#define _GNU_SOURCE
#include <errno.h>
#include <liburing.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>

typedef struct {
    struct io_uring ring;
    int registered_bufs;
    int sqpoll;
} m3tis_uring_t;

m3tis_uring_t* m3tis_uring_init(unsigned entries, int want_sqpoll, int* out_errno) {
    m3tis_uring_t* h = (m3tis_uring_t*)calloc(1, sizeof(m3tis_uring_t));
    if (!h) {
        if (out_errno) *out_errno = ENOMEM;
        return NULL;
    }
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    int ret = -EINVAL;

    if (want_sqpoll) {
        params.flags = IORING_SETUP_SQPOLL;
        params.sq_thread_idle = 2000; // 2ms idle poll before kernel thread sleeps
        ret = io_uring_queue_init_params(entries, &h->ring, &params);
        if (ret == 0) {
            h->sqpoll = 1;
            return h;
        }
        memset(&params, 0, sizeof(params));
    }

    // Modern low-overhead single-issuer cooperative taskrun (Linux 6.0+ / 7.2)
    params.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN;
    ret = io_uring_queue_init_params(entries, &h->ring, &params);
    if (ret != 0) {
        memset(&params, 0, sizeof(params));
        ret = io_uring_queue_init_params(entries, &h->ring, &params);
    }
    if (ret != 0) {
        if (out_errno) *out_errno = -ret;
        free(h);
        return NULL;
    }
    return h;
}

int m3tis_uring_is_sqpoll(const m3tis_uring_t* h) {
    return h ? h->sqpoll : 0;
}

void m3tis_uring_destroy(m3tis_uring_t* h) {
    if (!h) return;
    if (h->registered_bufs) {
        io_uring_unregister_buffers(&h->ring);
    }
    io_uring_queue_exit(&h->ring);
    free(h);
}

int m3tis_uring_register_buffers(m3tis_uring_t* h, void* const* ptrs, const size_t* lens, unsigned count) {
    if (!h || count == 0) return -EINVAL;
    struct iovec* iovs = (struct iovec*)calloc(count, sizeof(struct iovec));
    if (!iovs) return -ENOMEM;
    for (unsigned i = 0; i < count; ++i) {
        iovs[i].iov_base = ptrs[i];
        iovs[i].iov_len = lens[i];
    }
    int ret = io_uring_register_buffers(&h->ring, iovs, count);
    free(iovs);
    if (ret == 0) h->registered_bufs = 1;
    return ret;
}

int m3tis_uring_queue_read(
    m3tis_uring_t* h,
    int fd,
    void* buf,
    unsigned len,
    uint64_t offset,
    uint64_t user_data,
    int fixed_buf_idx
) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&h->ring);
    if (!sqe) return -EBUSY;
    if (fixed_buf_idx >= 0 && h->registered_bufs) {
        io_uring_prep_read_fixed(sqe, fd, buf, len, offset, fixed_buf_idx);
    } else {
        io_uring_prep_read(sqe, fd, buf, len, offset);
    }
    io_uring_sqe_set_data64(sqe, user_data);
    return 0;
}

int m3tis_uring_submit(m3tis_uring_t* h) {
    return io_uring_submit(&h->ring);
}

int m3tis_uring_submit_and_wait(m3tis_uring_t* h, unsigned wait_nr) {
    return io_uring_submit_and_wait(&h->ring, wait_nr);
}

// Reap up to `max_cqes` completions. If `wait_nr == 0`, strictly non-blocking user-space peek!
int m3tis_uring_reap(
    m3tis_uring_t* h,
    uint64_t* out_ud,
    int32_t* out_res,
    unsigned max_cqes,
    unsigned wait_nr
) {
    if (max_cqes == 0) return 0;
    struct io_uring_cqe* cqes[64];
    unsigned cap = max_cqes > 64 ? 64 : max_cqes;
    unsigned got = io_uring_peek_batch_cqe(&h->ring, cqes, cap);
    if (got == 0 && wait_nr > 0) {
        struct io_uring_cqe* first = NULL;
        int ret = io_uring_wait_cqe_nr(&h->ring, &first, wait_nr);
        if (ret < 0) return ret;
        got = io_uring_peek_batch_cqe(&h->ring, cqes, cap);
    }
    for (unsigned i = 0; i < got; ++i) {
        out_ud[i] = io_uring_cqe_get_data64(cqes[i]);
        out_res[i] = cqes[i]->res;
    }
    if (got > 0) {
        io_uring_cq_advance(&h->ring, got);
    }
    return (int)got;
}
