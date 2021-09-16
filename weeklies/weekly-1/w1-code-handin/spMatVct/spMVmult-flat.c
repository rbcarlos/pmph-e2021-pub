#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#endif
// Headers

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>


// Initialisation

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path);
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);

// Arrays

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
CUdeviceptr futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                      struct futhark_f32_1d *arr);
const int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                                    struct futhark_f32_1d *arr);
struct futhark_i64_1d ;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const
                                          int64_t *data, int64_t dim0);
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0);
int futhark_free_i64_1d(struct futhark_context *ctx,
                        struct futhark_i64_1d *arr);
int futhark_values_i64_1d(struct futhark_context *ctx,
                          struct futhark_i64_1d *arr, int64_t *data);
CUdeviceptr futhark_values_raw_i64_1d(struct futhark_context *ctx,
                                      struct futhark_i64_1d *arr);
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx,
                                    struct futhark_i64_1d *arr);

// Opaque values


// Entry points

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const
                       struct futhark_i64_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_i64_1d *in2, const
                       struct futhark_f32_1d *in3);

// Miscellaneous

int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_cuda
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}


static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                            const char *msg, ...) {
  if (errval) {
    char errnum[10];

    va_list vl;
    va_start(vl, msg);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, msg, vl);
    fprintf(stderr, " in %s() at line %d with error code %s\n",
            fun, line,
            sets_errno ? strerror(errno) : errnum);
    exit(errval);
  }
}

#define CHECK_ERR(err, msg...) check_err(err, 0, __func__, __LINE__, msg)
#define CHECK_ERRNO(err, msg...) check_err(err, 1, __func__, __LINE__, msg)

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  unsigned char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (unsigned char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

// End of util.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_REALTIME, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}

#endif

// End of timing.h.

#include <getopt.h>
#include <ctype.h>
#include <inttypes.h>
static const char *entry_point = "main";
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, const void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces(FILE *f) {
  int c;
  do {
    c = getc(f);
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, f);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(FILE *f, char *buf, int bufsize) {
 start:
  skipspaces(f);

  int i = 0;
  while (i < bufsize) {
    int c = getc(f);
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getc(f));
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, f);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(FILE *f, char *buf, int bufsize, const char* expected) {
  next_token(f, buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(FILE *f,
                                char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(f, buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(f, buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(FILE *f, char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(f, buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(f, buf, bufsize, "[")) {
      return 1;
    }

    next_token(f, buf, bufsize);

    if (sscanf(buf, "%"SCNu64, (uint64_t*)&shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(f, buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(f, buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(f, buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(FILE *f,
                          int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(f, buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(f, buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(f, buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNi8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNu8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(FILE *f, void* dest) {
  int num_elems_read = fread(dest, 1, 1, f);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary(FILE *f) {
  skipspaces(f);
  int c = getc(f);
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(f, &bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, f);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum(FILE *f) {
  char read_binname[4];

  int num_matched = fscanf(f, "%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(FILE *f, const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum(f);
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(FILE *f,
                          const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum(f);
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, f);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, f);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(FILE *f, const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary(f)) {
    return read_str_array(f, expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(f, expected_type, data, shape, dims);
  }
}

static int end_of_input(FILE *f) {
  skipspaces(f);
  char token[2];
  next_token(f, token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      fprintf(out, "empty(");
      for (int64_t i = 0; i < rank; i++) {
        fprintf(out, "[%"PRIi64"]", shape[i]);
      }
      fprintf(out, "%s", elem_type->type_name);
      fprintf(out, ")");
    } else if (rank==1) {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    } else {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    }
  }
  return 0;
}

static int write_bin_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fwrite(elem_type->binname, 4, 1, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      const unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type,
                       const void *data,
                       const int64_t *shape,
                       const int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(FILE *f,
                       const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary(f)) {
    char buf[100];
    next_token(f, buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(f, expected_type);
    int64_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, (size_t)elem_size, 1, f);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

// Start of server.h.

// Forward declarations of things that we technically don't know until
// the application header file is included, but which we need.
struct futhark_context;
char *futhark_context_get_error(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);

typedef int (*restore_fn)(const void*, FILE *, struct futhark_context*, void*);
typedef void (*store_fn)(const void*, FILE *, struct futhark_context*, void*);
typedef int (*free_fn)(const void*, struct futhark_context*, void*);

struct type {
  const char *name;
  restore_fn restore;
  store_fn store;
  free_fn free;
  const void *aux;
};

int free_scalar(const void *aux, struct futhark_context *ctx, void *p) {
  (void)aux;
  (void)ctx;
  (void)p;
  // Nothing to do.
  return 0;
}

#define DEF_SCALAR_TYPE(T)                                              \
  int restore_##T(const void *aux, FILE *f,                             \
                  struct futhark_context *ctx, void *p) {               \
    (void)aux;                                                          \
    (void)ctx;                                                          \
    return read_scalar(f, &T##_info, p);                                \
  }                                                                     \
                                                                        \
  void store_##T(const void *aux, FILE *f,                              \
                 struct futhark_context *ctx, void *p) {                \
    (void)aux;                                                          \
    (void)ctx;                                                          \
    write_scalar(f, 1, &T##_info, p);                                   \
  }                                                                     \
                                                                        \
  struct type type_##T =                                                \
    { .name = #T,                                                       \
      .restore = restore_##T,                                           \
      .store = store_##T,                                               \
      .free = free_scalar                                               \
    }                                                                   \

DEF_SCALAR_TYPE(i8);
DEF_SCALAR_TYPE(i16);
DEF_SCALAR_TYPE(i32);
DEF_SCALAR_TYPE(i64);
DEF_SCALAR_TYPE(u8);
DEF_SCALAR_TYPE(u16);
DEF_SCALAR_TYPE(u32);
DEF_SCALAR_TYPE(u64);
DEF_SCALAR_TYPE(f32);
DEF_SCALAR_TYPE(f64);
DEF_SCALAR_TYPE(bool);

struct value {
  struct type *type;
  union {
    void *v_ptr;
    int8_t  v_i8;
    int16_t v_i16;
    int32_t v_i32;
    int64_t v_i64;

    uint8_t  v_u8;
    uint16_t v_u16;
    uint32_t v_u32;
    uint64_t v_u64;

    float v_f32;
    double v_f64;

    bool v_bool;
  } value;
};

void* value_ptr(struct value *v) {
  if (v->type == &type_i8) {
    return &v->value.v_i8;
  }
  if (v->type == &type_i16) {
    return &v->value.v_i16;
  }
  if (v->type == &type_i32) {
    return &v->value.v_i32;
  }
  if (v->type == &type_i64) {
    return &v->value.v_i64;
  }
  if (v->type == &type_u8) {
    return &v->value.v_u8;
  }
  if (v->type == &type_u16) {
    return &v->value.v_u16;
  }
  if (v->type == &type_u32) {
    return &v->value.v_u32;
  }
  if (v->type == &type_u64) {
    return &v->value.v_u64;
  }
  if (v->type == &type_f32) {
    return &v->value.v_f32;
  }
  if (v->type == &type_f64) {
    return &v->value.v_f64;
  }
  if (v->type == &type_bool) {
    return &v->value.v_bool;
  }
  return &v->value.v_ptr;
}

struct variable {
  // NULL name indicates free slot.  Name is owned by this struct.
  char *name;
  struct value value;
};

typedef int (*entry_point_fn)(struct futhark_context*, void**, void**);

struct entry_point {
  const char *name;
  entry_point_fn f;
  struct type **out_types;
  struct type **in_types;
};

int entry_num_ins(struct entry_point *e) {
  int count = 0;
  while (e->in_types[count]) {
    count++;
  }
  return count;
}

int entry_num_outs(struct entry_point *e) {
  int count = 0;
  while (e->out_types[count]) {
    count++;
  }
  return count;
}

struct futhark_prog {
  // Last entry point identified by NULL name.
  struct entry_point *entry_points;
  // Last type identified by NULL name.
  struct type **types;
};

struct server_state {
  struct futhark_prog prog;
  struct futhark_context *ctx;
  int variables_capacity;
  struct variable *variables;
};

struct variable* get_variable(struct server_state *s,
                              const char *name) {
  for (int i = 0; i < s->variables_capacity; i++) {
    if (s->variables[i].name != NULL &&
        strcmp(s->variables[i].name, name) == 0) {
      return &s->variables[i];
    }
  }

  return NULL;
}

struct variable* create_variable(struct server_state *s,
                                 const char *name,
                                 struct type *type) {
  int found = -1;
  for (int i = 0; i < s->variables_capacity; i++) {
    if (found == -1 && s->variables[i].name == NULL) {
      found = i;
    } else if (s->variables[i].name != NULL &&
               strcmp(s->variables[i].name, name) == 0) {
      return NULL;
    }
  }

  if (found != -1) {
    // Found a free spot.
    s->variables[found].name = strdup(name);
    s->variables[found].value.type = type;
    return &s->variables[found];
  }

  // Need to grow the buffer.
  found = s->variables_capacity;
  s->variables_capacity *= 2;
  s->variables = realloc(s->variables,
                         s->variables_capacity * sizeof(struct variable));

  s->variables[found].name = strdup(name);
  s->variables[found].value.type = type;

  for (int i = found+1; i < s->variables_capacity; i++) {
    s->variables[i].name = NULL;
  }

  return &s->variables[found];
}

void drop_variable(struct variable *v) {
  free(v->name);
  v->name = NULL;
}

int arg_exists(const char *args[], int i) {
  return args[i] != NULL;
}

const char* get_arg(const char *args[], int i) {
  if (!arg_exists(args, i)) {
    futhark_panic(1, "Insufficient command args.\n");
  }
  return args[i];
}

struct type* get_type(struct server_state *s, const char *name) {
  for (int i = 0; s->prog.types[i]; i++) {
    if (strcmp(s->prog.types[i]->name, name) == 0) {
      return s->prog.types[i];
    }
  }

  futhark_panic(1, "Unknown type %s\n", name);
  return NULL;
}

struct entry_point* get_entry_point(struct server_state *s, const char *name) {
  for (int i = 0; s->prog.entry_points[i].name; i++) {
    if (strcmp(s->prog.entry_points[i].name, name) == 0) {
      return &s->prog.entry_points[i];
    }
  }

  return NULL;
}

// Print the command-done marker, indicating that we are ready for
// more input.
void ok() {
  printf("%%%%%% OK\n");
  fflush(stdout);
}

// Print the failure marker.  Output is now an error message until the
// next ok().
void failure() {
  printf("%%%%%% FAILURE\n");
}

void error_check(struct server_state *s, int err) {
  if (err != 0) {
    failure();
    char *error = futhark_context_get_error(s->ctx);
    puts(error);
    free(error);
  }
}

void cmd_call(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);

  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_outs = entry_num_outs(e);
  int num_ins = entry_num_ins(e);
  void* outs[num_outs];
  void* ins[num_ins];

  for (int i = 0; i < num_ins; i++) {
    const char *in_name = get_arg(args, 1+num_outs+i);
    struct variable *v = get_variable(s, in_name);
    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", in_name);
      return;
    }
    if (v->value.type != e->in_types[i]) {
      failure();
      printf("Wrong input type.  Expected %s, got %s.\n",
             e->in_types[i]->name, v->value.type->name);
      return;
    }
    ins[i] = value_ptr(&v->value);
  }

  for (int i = 0; i < num_outs; i++) {
    const char *out_name = get_arg(args, 1+i);
    struct variable *v = create_variable(s, out_name, e->out_types[i]);
    if (v == NULL) {
      failure();
      printf("Variable already exists: %s\n", out_name);
      return;
    }
    outs[i] = value_ptr(&v->value);
  }

  int64_t t_start = get_wall_time();
  int err = e->f(s->ctx, outs, ins);
  err |= futhark_context_sync(s->ctx);
  int64_t t_end = get_wall_time();
  long long int elapsed_usec = t_end - t_start;
  printf("runtime: %lld\n", elapsed_usec);

  error_check(s, err);
  if (err != 0) {
    // Need to uncreate the output variables, which would otherwise be left
    // in an uninitialised state.
    for (int i = 0; i < num_outs; i++) {
      const char *out_name = get_arg(args, 1+i);
      struct variable *v = get_variable(s, out_name);
      if (v) {
        drop_variable(v);
      }
    }
  }
}

void cmd_restore(struct server_state *s, const char *args[]) {
  const char *fname = get_arg(args, 0);

  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    failure();
    printf("Failed to open %s: %s\n", fname, strerror(errno));
  } else {
    int values = 0;
    for (int i = 1; arg_exists(args, i); i+=2, values++) {
      const char *vname = get_arg(args, i);
      const char *type = get_arg(args, i+1);

      struct type *t = get_type(s, type);
      struct variable *v = create_variable(s, vname, t);

      if (v == NULL) {
        failure();
        printf("Variable already exists: %s\n", vname);
        return;
      }

      if (t->restore(t->aux, f, s->ctx, value_ptr(&v->value)) != 0) {
        failure();
        printf("Failed to restore variable %s.\n"
               "Possibly malformed data in %s (errno: %s)\n",
               vname, fname, strerror(errno));
        drop_variable(v);
        break;
      }
    }

    if (end_of_input(f) != 0) {
      failure();
      printf("Expected EOF after reading %d values from %s\n",
             values, fname);
    }

    fclose(f);
  }

  int err = futhark_context_sync(s->ctx);
  error_check(s, err);
}

void cmd_store(struct server_state *s, const char *args[]) {
  const char *fname = get_arg(args, 0);

  FILE *f = fopen(fname, "wb");
  if (f == NULL) {
    failure();
    printf("Failed to open %s: %s\n", fname, strerror(errno));
  } else {
    for (int i = 1; arg_exists(args, i); i++) {
      const char *vname = get_arg(args, i);
      struct variable *v = get_variable(s, vname);

      if (v == NULL) {
        failure();
        printf("Unknown variable: %s\n", vname);
        return;
      }

      struct type *t = v->value.type;
      t->store(t->aux, f, s->ctx, value_ptr(&v->value));
    }
    fclose(f);
  }
}

void cmd_free(struct server_state *s, const char *args[]) {
  for (int i = 0; arg_exists(args, i); i++) {
    const char *name = get_arg(args, i);
    struct variable *v = get_variable(s, name);

    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", name);
      return;
    }

    struct type *t = v->value.type;

    int err = t->free(t->aux, s->ctx, value_ptr(&v->value));
    error_check(s, err);
    drop_variable(v);
  }
}

void cmd_inputs(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_ins = entry_num_ins(e);
  for (int i = 0; i < num_ins; i++) {
    puts(e->in_types[i]->name);
  }
}

void cmd_outputs(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_outs = entry_num_outs(e);
  for (int i = 0; i < num_outs; i++) {
    puts(e->out_types[i]->name);
  }
}

void cmd_clear(struct server_state *s, const char *args[]) {
  (void)args;
  int err = 0;
  for (int i = 0; i < s->variables_capacity; i++) {
    struct variable *v = &s->variables[i];
    if (v->name != NULL) {
      err |= v->value.type->free(v->value.type->aux, s->ctx, value_ptr(&v->value));
      drop_variable(v);
    }
  }
  err |= futhark_context_clear_caches(s->ctx);
  error_check(s, err);
}

void cmd_pause_profiling(struct server_state *s, const char *args[]) {
  (void)args;
  futhark_context_pause_profiling(s->ctx);
}

void cmd_unpause_profiling(struct server_state *s, const char *args[]) {
  (void)args;
  futhark_context_unpause_profiling(s->ctx);
}

void cmd_report(struct server_state *s, const char *args[]) {
  (void)args;
  char *report = futhark_context_report(s->ctx);
  puts(report);
  free(report);
}

void process_line(struct server_state *s, char *line) {
  int max_num_tokens = 100;
  const char* tokens[max_num_tokens];
  int num_tokens = 0;
  char *saveptr;

  char *tmp = line;
  while ((tokens[num_tokens] = strtok_r(tmp, " \n", &saveptr)) != NULL) {
    num_tokens++;
    if (num_tokens == max_num_tokens) {
      futhark_panic(1, "Line too long.\n");
    }
    tmp = NULL;
  }

  const char *command = tokens[0];

  if (command == NULL) {
    failure();
    printf("Empty line\n");
  } else if (strcmp(command, "call") == 0) {
    cmd_call(s, tokens+1);
  } else if (strcmp(command, "restore") == 0) {
    cmd_restore(s, tokens+1);
  } else if (strcmp(command, "store") == 0) {
    cmd_store(s, tokens+1);
  } else if (strcmp(command, "free") == 0) {
    cmd_free(s, tokens+1);
  } else if (strcmp(command, "inputs") == 0) {
    cmd_inputs(s, tokens+1);
  } else if (strcmp(command, "outputs") == 0) {
    cmd_outputs(s, tokens+1);
  } else if (strcmp(command, "clear") == 0) {
    cmd_clear(s, tokens+1);
  } else if (strcmp(command, "pause_profiling") == 0) {
    cmd_pause_profiling(s, tokens+1);
  } else if (strcmp(command, "unpause_profiling") == 0) {
    cmd_unpause_profiling(s, tokens+1);
  } else if (strcmp(command, "report") == 0) {
    cmd_report(s, tokens+1);
  } else {
    futhark_panic(1, "Unknown command: %s\n", command);
  }
  ok();
}

void run_server(struct futhark_prog *prog, struct futhark_context *ctx) {
  char *line = NULL;
  size_t buflen = 0;
  ssize_t linelen;

  struct server_state s = {
    .ctx = ctx,
    .variables_capacity = 100,
    .prog = *prog
  };

  s.variables = malloc(s.variables_capacity * sizeof(struct variable));

  for (int i = 0; i < s.variables_capacity; i++) {
    s.variables[i].name = NULL;
  }

  while ((linelen = getline(&line, &buflen, stdin)) > 0) {
    process_line(&s, line);
  }

  free(line);
}

// The aux struct lets us write generic method implementations without
// code duplication.

typedef void* (*array_new_fn)(struct futhark_context *, const void*, const int64_t*);
typedef const int64_t* (*array_shape_fn)(struct futhark_context*, void*);
typedef int (*array_values_fn)(struct futhark_context*, void*, void*);
typedef int (*array_free_fn)(struct futhark_context*, void*);

struct array_aux {
  int rank;
  const struct primtype_info_t* info;
  const char *name;
  array_new_fn new;
  array_shape_fn shape;
  array_values_fn values;
  array_free_fn free;
};

int restore_array(const struct array_aux *aux, FILE *f,
                  struct futhark_context *ctx, void *p) {
  void *data = NULL;
  int64_t shape[aux->rank];
  if (read_array(f, aux->info, &data, shape, aux->rank) != 0) {
    return 1;
  }

  void *arr = aux->new(ctx, data, shape);
  if (arr == NULL) {
    return 1;
  }

  *(void**)p = arr;
  free(data);
  return 0;
}

void store_array(const struct array_aux *aux, FILE *f,
                 struct futhark_context *ctx, void *p) {
  void *arr = *(void**)p;
  const int64_t *shape = aux->shape(ctx, arr);
  int64_t size = sizeof(aux->info->size);
  for (int i = 0; i < aux->rank; i++) {
    size *= shape[i];
  }
  int32_t *data = malloc(size);
  assert(aux->values(ctx, arr, data) == 0);
  assert(futhark_context_sync(ctx) == 0);
  assert(write_array(f, 1, aux->info, data, shape, aux->rank) == 0);
  free(data);
}

int free_array(const struct array_aux *aux,
                struct futhark_context *ctx, void *p) {
  void *arr = *(void**)p;
  return aux->free(ctx, arr);
}

typedef int (*opaque_free_fn)(struct futhark_context*, void*);

struct opaque_aux {
  opaque_free_fn free;
};

int free_opaque(const struct opaque_aux *aux,
                 struct futhark_context *ctx, void *p) {
  void *obj = *(void**)p;
  return aux->free(ctx, obj);
}

// End of server.h.

// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

void *futhark_new_f32_1d_wrap(struct futhark_context *ctx, const void *p, const
                              int64_t *shape)
{
    return futhark_new_f32_1d(ctx, p, shape[0]);
}
struct array_aux type_f32_1d_aux = {.name ="[]f32", .rank =1, .info =&f32_info,
                                    .new =
                                    (array_new_fn) futhark_new_f32_1d_wrap,
                                    .free =(array_free_fn) futhark_free_f32_1d,
                                    .shape =
                                    (array_shape_fn) futhark_shape_f32_1d,
                                    .values =
                                    (array_values_fn) futhark_values_f32_1d};
struct type type_f32_1d = {.name ="[]f32", .restore =(restore_fn) restore_array,
                           .store =(store_fn) store_array, .free =
                           (free_fn) free_array, .aux =&type_f32_1d_aux};
void *futhark_new_i64_1d_wrap(struct futhark_context *ctx, const void *p, const
                              int64_t *shape)
{
    return futhark_new_i64_1d(ctx, p, shape[0]);
}
struct array_aux type_i64_1d_aux = {.name ="[]i64", .rank =1, .info =&i64_info,
                                    .new =
                                    (array_new_fn) futhark_new_i64_1d_wrap,
                                    .free =(array_free_fn) futhark_free_i64_1d,
                                    .shape =
                                    (array_shape_fn) futhark_shape_i64_1d,
                                    .values =
                                    (array_values_fn) futhark_values_i64_1d};
struct type type_i64_1d = {.name ="[]i64", .restore =(restore_fn) restore_array,
                           .store =(store_fn) store_array, .free =
                           (free_fn) free_array, .aux =&type_i64_1d_aux};
struct type *main_out_types[] = {&type_f32_1d, NULL};
struct type *main_in_types[] = {&type_i64_1d, &type_f32_1d, &type_i64_1d,
                                &type_f32_1d, NULL};
int call_main(struct futhark_context *ctx, void **outs, void **ins)
{
    struct futhark_f32_1d **out0 = outs[0];
    struct futhark_i64_1d *in0 = *(struct futhark_i64_1d **) ins[0];
    struct futhark_f32_1d *in1 = *(struct futhark_f32_1d **) ins[1];
    struct futhark_i64_1d *in2 = *(struct futhark_i64_1d **) ins[2];
    struct futhark_f32_1d *in3 = *(struct futhark_f32_1d **) ins[3];
    
    return futhark_entry_main(ctx, out0, in0, in1, in2, in3);
}
struct type *types[] = {&type_f32_1d, &type_i64_1d, NULL};
struct entry_point entry_points[] = {{.name ="main", .f =call_main, .in_types =
                                      main_in_types, .out_types =
                                      main_out_types}, {.name =NULL}};
struct futhark_prog prog = {.types =types, .entry_points =entry_points};
int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"debugging", no_argument, NULL, 1},
                                           {"log", no_argument, NULL, 2},
                                           {"help", no_argument, NULL, 3},
                                           {"device", required_argument, NULL,
                                            4}, {"default-group-size",
                                                 required_argument, NULL, 5},
                                           {"default-num-groups",
                                            required_argument, NULL, 6},
                                           {"default-tile-size",
                                            required_argument, NULL, 7},
                                           {"default-threshold",
                                            required_argument, NULL, 8},
                                           {"print-sizes", no_argument, NULL,
                                            9}, {"size", required_argument,
                                                 NULL, 10}, {"tuning",
                                                             required_argument,
                                                             NULL, 11},
                                           {"dump-cuda", required_argument,
                                            NULL, 12}, {"load-cuda",
                                                        required_argument, NULL,
                                                        13}, {"dump-ptx",
                                                              required_argument,
                                                              NULL, 14},
                                           {"load-ptx", required_argument, NULL,
                                            15}, {"nvrtc-option",
                                                  required_argument, NULL, 16},
                                           {"profile", no_argument, NULL, 17},
                                           {0, 0, 0, 0}};
    static char *option_descriptions =
                "  -D/--debugging           Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                 Print various low-overhead logging information while running.\n  -h/--help                Print help information and exit.\n  -d/--device NAME         Use the first OpenCL device whose name contains the given string.\n  --default-group-size INT The default size of OpenCL workgroups that are launched.\n  --default-num-groups INT The default number of OpenCL workgroups that are launched.\n  --default-tile-size INT  The default tile size used when performing two-dimensional tiling.\n  --default-threshold INT  The default parallelism threshold.\n  --print-sizes            Print all sizes that can be set with -size or --tuning.\n  --size ASSIGNMENT        Set a configurable run-time parameter to the given value.\n  --tuning FILE            Read size=value assignments from the given file.\n  --dump-cuda FILE         Dump the embedded CUDA kernels to the indicated file.\n  --load-cuda FILE         Instead of using the embedded CUDA kernels, load them from the indicated file.\n  --dump-ptx FILE          Dump the PTX-compiled version of the embedded kernels to the indicated file.\n  --load-ptx FILE          Load PTX code from the indicated file.\n  --nvrtc-option OPT       Add an additional build option to the string passed to NVRTC.\n  -P/--profile             Gather profiling data while executing and print out a summary at the end.\n";
    
    while ((ch = getopt_long(argc, argv, ":DLhd:P", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 2 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 3 || ch == 'h') {
            printf("Usage: %s [OPTION]...\nOptions:\n\n%s\nFor more information, consult the Futhark User's Guide or the man pages.\n",
                   fut_progname, option_descriptions);
            exit(0);
        }
        if (ch == 4 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 5)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 6)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 7)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 8)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 9) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 10) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    futhark_panic(1, "Unknown size: %s\n", name);
            } else
                futhark_panic(1, "Invalid argument for size option: %s\n",
                              optarg);
        }
        if (ch == 11) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                futhark_panic(1, "When loading tuning from '%s': %s\n", optarg,
                              ret);
        }
        if (ch == 12) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 13)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 14) {
            futhark_context_config_dump_ptx_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 15)
            futhark_context_config_load_ptx_from(cfg, optarg);
        if (ch == 16)
            futhark_context_config_add_nvrtc_option(cfg, optarg);
        if (ch == 17 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind -
                                                                       1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "  -D/--debugging           Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                 Print various low-overhead logging information while running.\n  -h/--help                Print help information and exit.\n  -d/--device NAME         Use the first OpenCL device whose name contains the given string.\n  --default-group-size INT The default size of OpenCL workgroups that are launched.\n  --default-num-groups INT The default number of OpenCL workgroups that are launched.\n  --default-tile-size INT  The default tile size used when performing two-dimensional tiling.\n  --default-threshold INT  The default parallelism threshold.\n  --print-sizes            Print all sizes that can be set with -size or --tuning.\n  --size ASSIGNMENT        Set a configurable run-time parameter to the given value.\n  --tuning FILE            Read size=value assignments from the given file.\n  --dump-cuda FILE         Dump the embedded CUDA kernels to the indicated file.\n  --load-cuda FILE         Instead of using the embedded CUDA kernels, load them from the indicated file.\n  --dump-ptx FILE          Dump the PTX-compiled version of the embedded kernels to the indicated file.\n  --load-ptx FILE          Load PTX code from the indicated file.\n  --nvrtc-option OPT       Add an additional build option to the string passed to NVRTC.\n  -P/--profile             Gather profiling data while executing and print out a summary at the end.\n");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    futhark_context_set_logging_file(ctx, stdout);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "Error during context initialisation:\n%s", error);
    if (entry_point != NULL)
        run_server(&prog, ctx);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.

static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzll(x);
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
static int init_constants(struct futhark_context *);
static int free_constants(struct futhark_context *);
static int32_t mainziid_counter_mem_realtype_5164[1];
static int32_t mainziid_counter_mem_realtype_5181[1];
struct memblock_device {
    int *references;
    CUdeviceptr mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
#include <cuda.h>
#include <nvrtc.h>
typedef CUdeviceptr fl_mem_t;
// Start of free_list.h.

// An entry in the free list.  May be invalid, to avoid having to
// deallocate entries as soon as they are removed.  There is also a
// tag, to help with memory reuse.
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

// Remove invalid entries from the free list.
static void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }

  // Now p is the number of used elements.  We don't want it to go
  // less than the default capacity (although in practice it's OK as
  // long as it doesn't become 1).
  if (p < 30) {
    p = 30;
  }
  l->entries = realloc(l->entries, p * sizeof(struct free_list_entry));
  l->capacity = p;
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

// Find and remove a memory block of the indicated tag, or if that
// does not exist, another memory block with exactly the desired size.
// Returns 0 on success.
static int free_list_find(struct free_list *l, size_t size,
                          size_t *size_out, fl_mem_t *mem_out) {
  int size_match = -1;
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid &&
        size <= l->entries[i].size &&
        (size_match < 0 || l->entries[i].size < l->entries[size_match].size)) {
      // If this entry is valid, has sufficient size, and is smaller than the
      // best entry found so far, use this entry.
      size_match = i;
    }
  }

  if (size_match >= 0) {
    l->entries[size_match].valid = 0;
    *size_out = l->entries[size_match].size;
    *mem_out = l->entries[size_match].mem;
    l->used--;
    return 0;
  } else {
    return 1;
  }
}

// Remove the first block in the free list.  Returns 0 if a block was
// removed, and nonzero if the free list was already empty.
static int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

// End of free_list.h.

// Start of cuda.h.

#define CUDA_SUCCEED(x) cuda_api_succeed(x, #x, __FILE__, __LINE__)
#define NVRTC_SUCCEED(x) nvrtc_api_succeed(x, #x, __FILE__, __LINE__)

static inline void cuda_api_succeed(CUresult res, const char *call,
    const char *file, int line) {
  if (res != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(res, &err_str);
    if (err_str == NULL) { err_str = "Unknown"; }
    futhark_panic(-1, "%s:%d: CUDA call\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, err_str);
  }
}

static inline void nvrtc_api_succeed(nvrtcResult res, const char *call,
                                     const char *file, int line) {
  if (res != NVRTC_SUCCESS) {
    const char *err_str = nvrtcGetErrorString(res);
    futhark_panic(-1, "%s:%d: NVRTC call\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, err_str);
  }
}

struct cuda_config {
  int debugging;
  int logging;
  const char *preferred_device;
  int preferred_device_num;

  const char *dump_program_to;
  const char *load_program_from;

  const char *dump_ptx_to;
  const char *load_ptx_from;

  size_t default_block_size;
  size_t default_grid_size;
  size_t default_tile_size;
  size_t default_threshold;

  int default_block_size_changed;
  int default_grid_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  int64_t *size_values;
  const char **size_classes;
};

static void cuda_config_init(struct cuda_config *cfg,
                             int num_sizes,
                             const char *size_names[],
                             const char *size_vars[],
                             int64_t *size_values,
                             const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_device = "";
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;

  cfg->dump_ptx_to = NULL;
  cfg->load_ptx_from = NULL;

  cfg->default_block_size = 256;
  cfg->default_grid_size = 0; // Set properly later.
  cfg->default_tile_size = 32;
  cfg->default_threshold = 32*1024;

  cfg->default_block_size_changed = 0;
  cfg->default_grid_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

// A record of something that happened.
struct profiling_record {
  cudaEvent_t *events; // Points to two events.
  int *runs;
  int64_t *runtime;
};

struct cuda_context {
  CUdevice dev;
  CUcontext cu_ctx;
  CUmodule module;

  struct cuda_config cfg;

  struct free_list free_list;

  size_t max_block_size;
  size_t max_grid_size;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_shared_memory;
  size_t max_bespoke;

  size_t lockstep_width;

  struct profiling_record *profiling_records;
  int profiling_records_capacity;
  int profiling_records_used;
};

#define CU_DEV_ATTR(x) (CU_DEVICE_ATTRIBUTE_##x)
#define device_query(dev,attrib) _device_query(dev, CU_DEV_ATTR(attrib))
static int _device_query(CUdevice dev, CUdevice_attribute attrib) {
  int val;
  CUDA_SUCCEED(cuDeviceGetAttribute(&val, attrib, dev));
  return val;
}

#define CU_FUN_ATTR(x) (CU_FUNC_ATTRIBUTE_##x)
#define function_query(fn,attrib) _function_query(dev, CU_FUN_ATTR(attrib))
static int _function_query(CUfunction dev, CUfunction_attribute attrib) {
  int val;
  CUDA_SUCCEED(cuFuncGetAttribute(&val, attrib, dev));
  return val;
}

static void set_preferred_device(struct cuda_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
}

static int cuda_device_setup(struct cuda_context *ctx) {
  char name[256];
  int count, chosen = -1, best_cc = -1;
  int cc_major_best, cc_minor_best;
  int cc_major, cc_minor;
  CUdevice dev;

  CUDA_SUCCEED(cuDeviceGetCount(&count));
  if (count == 0) { return 1; }

  int num_device_matches = 0;

  // XXX: Current device selection policy is to choose the device with the
  // highest compute capability (if no preferred device is set).
  // This should maybe be changed, since greater compute capability is not
  // necessarily an indicator of better performance.
  for (int i = 0; i < count; i++) {
    CUDA_SUCCEED(cuDeviceGet(&dev, i));

    cc_major = device_query(dev, COMPUTE_CAPABILITY_MAJOR);
    cc_minor = device_query(dev, COMPUTE_CAPABILITY_MINOR);

    CUDA_SUCCEED(cuDeviceGetName(name, sizeof(name) - 1, dev));
    name[sizeof(name) - 1] = 0;

    if (ctx->cfg.debugging) {
      fprintf(stderr, "Device #%d: name=\"%s\", compute capability=%d.%d\n",
          i, name, cc_major, cc_minor);
    }

    if (device_query(dev, COMPUTE_MODE) == CU_COMPUTEMODE_PROHIBITED) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Device #%d is compute-prohibited, ignoring\n", i);
      }
      continue;
    }

    if (best_cc == -1 || cc_major > cc_major_best ||
        (cc_major == cc_major_best && cc_minor > cc_minor_best)) {
      best_cc = i;
      cc_major_best = cc_major;
      cc_minor_best = cc_minor;
    }

    if (strstr(name, ctx->cfg.preferred_device) != NULL &&
        num_device_matches++ == ctx->cfg.preferred_device_num) {
      chosen = i;
      break;
    }
  }

  if (chosen == -1) { chosen = best_cc; }
  if (chosen == -1) { return 1; }

  if (ctx->cfg.debugging) {
    fprintf(stderr, "Using device #%d\n", chosen);
  }

  CUDA_SUCCEED(cuDeviceGet(&ctx->dev, chosen));
  return 0;
}

static char *concat_fragments(const char *src_fragments[]) {
  size_t src_len = 0;
  const char **p;

  for (p = src_fragments; *p; p++) {
    src_len += strlen(*p);
  }

  char *src = (char*) malloc(src_len + 1);
  size_t n = 0;
  for (p = src_fragments; *p; p++) {
    strcpy(src + n, *p);
    n += strlen(*p);
  }

  return src;
}

static const char *cuda_nvrtc_get_arch(CUdevice dev) {
  struct {
    int major;
    int minor;
    const char *arch_str;
  } static const x[] = {
    { 3, 0, "compute_30" },
    { 3, 2, "compute_32" },
    { 3, 5, "compute_35" },
    { 3, 7, "compute_37" },
    { 5, 0, "compute_50" },
    { 5, 2, "compute_52" },
    { 5, 3, "compute_53" },
    { 6, 0, "compute_60" },
    { 6, 1, "compute_61" },
    { 6, 2, "compute_62" },
    { 7, 0, "compute_70" },
    { 7, 2, "compute_72" },
    { 7, 5, "compute_75" }
  };

  int major = device_query(dev, COMPUTE_CAPABILITY_MAJOR);
  int minor = device_query(dev, COMPUTE_CAPABILITY_MINOR);

  int chosen = -1;
  for (int i = 0; i < sizeof(x)/sizeof(x[0]); i++) {
    if (x[i].major < major || (x[i].major == major && x[i].minor <= minor)) {
      chosen = i;
    } else {
      break;
    }
  }

  if (chosen == -1) {
    futhark_panic(-1, "Unsupported compute capability %d.%d\n", major, minor);
  }

  if (x[chosen].major != major || x[chosen].minor != minor) {
    fprintf(stderr,
            "Warning: device compute capability is %d.%d, but newest supported by Futhark is %d.%d.\n",
            major, minor, x[chosen].major, x[chosen].minor);
  }

  return x[chosen].arch_str;
}

static char *cuda_nvrtc_build(struct cuda_context *ctx, const char *src,
                              const char *extra_opts[]) {
  nvrtcProgram prog;
  NVRTC_SUCCEED(nvrtcCreateProgram(&prog, src, "futhark-cuda", 0, NULL, NULL));
  int arch_set = 0, num_extra_opts;

  // nvrtc cannot handle multiple -arch options.  Hence, if one of the
  // extra_opts is -arch, we have to be careful not to do our usual
  // automatic generation.
  for (num_extra_opts = 0; extra_opts[num_extra_opts] != NULL; num_extra_opts++) {
    if (strstr(extra_opts[num_extra_opts], "-arch")
        == extra_opts[num_extra_opts] ||
        strstr(extra_opts[num_extra_opts], "--gpu-architecture")
        == extra_opts[num_extra_opts]) {
      arch_set = 1;
    }
  }

  size_t n_opts, i = 0, i_dyn, n_opts_alloc = 20 + num_extra_opts + ctx->cfg.num_sizes;
  const char **opts = (const char**) malloc(n_opts_alloc * sizeof(const char *));
  if (!arch_set) {
    opts[i++] = "-arch";
    opts[i++] = cuda_nvrtc_get_arch(ctx->dev);
  }
  opts[i++] = "-default-device";
  if (ctx->cfg.debugging) {
    opts[i++] = "-G";
    opts[i++] = "-lineinfo";
  } else {
    opts[i++] = "--disable-warnings";
  }
  i_dyn = i;
  for (size_t j = 0; j < ctx->cfg.num_sizes; j++) {
    opts[i++] = msgprintf("-D%s=%zu", ctx->cfg.size_vars[j],
        ctx->cfg.size_values[j]);
  }
  opts[i++] = msgprintf("-DLOCKSTEP_WIDTH=%zu", ctx->lockstep_width);
  opts[i++] = msgprintf("-DMAX_THREADS_PER_BLOCK=%zu", ctx->max_block_size);

  // It is crucial that the extra_opts are last, so that the free()
  // logic below does not cause problems.
  for (int j = 0; extra_opts[j] != NULL; j++) {
    opts[i++] = extra_opts[j];
  }

  n_opts = i;

  if (ctx->cfg.debugging) {
    fprintf(stderr, "NVRTC compile options:\n");
    for (size_t j = 0; j < n_opts; j++) {
      fprintf(stderr, "\t%s\n", opts[j]);
    }
    fprintf(stderr, "\n");
  }

  nvrtcResult res = nvrtcCompileProgram(prog, n_opts, opts);
  if (res != NVRTC_SUCCESS) {
    size_t log_size;
    if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS) {
      char *log = (char*) malloc(log_size);
      if (nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS) {
        fprintf(stderr,"Compilation log:\n%s\n", log);
      }
      free(log);
    }
    NVRTC_SUCCEED(res);
  }

  for (i = i_dyn; i < n_opts-num_extra_opts; i++) { free((char *)opts[i]); }
  free(opts);

  char *ptx;
  size_t ptx_size;
  NVRTC_SUCCEED(nvrtcGetPTXSize(prog, &ptx_size));
  ptx = (char*) malloc(ptx_size);
  NVRTC_SUCCEED(nvrtcGetPTX(prog, ptx));

  NVRTC_SUCCEED(nvrtcDestroyProgram(&prog));

  return ptx;
}

static void cuda_size_setup(struct cuda_context *ctx)
{
  if (ctx->cfg.default_block_size > ctx->max_block_size) {
    if (ctx->cfg.default_block_size_changed) {
      fprintf(stderr,
          "Note: Device limits default block size to %zu (down from %zu).\n",
          ctx->max_block_size, ctx->cfg.default_block_size);
    }
    ctx->cfg.default_block_size = ctx->max_block_size;
  }
  if (ctx->cfg.default_grid_size > ctx->max_grid_size) {
    if (ctx->cfg.default_grid_size_changed) {
      fprintf(stderr,
          "Note: Device limits default grid size to %zu (down from %zu).\n",
          ctx->max_grid_size, ctx->cfg.default_grid_size);
    }
    ctx->cfg.default_grid_size = ctx->max_grid_size;
  }
  if (ctx->cfg.default_tile_size > ctx->max_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr,
          "Note: Device limits default tile size to %zu (down from %zu).\n",
          ctx->max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = ctx->max_tile_size;
  }

  if (!ctx->cfg.default_grid_size_changed) {
    ctx->cfg.default_grid_size =
      (device_query(ctx->dev, MULTIPROCESSOR_COUNT) *
       device_query(ctx->dev, MAX_THREADS_PER_MULTIPROCESSOR))
      / ctx->cfg.default_block_size;
  }

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    int64_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    int64_t max_value = 0, default_value = 0;

    if (strstr(size_class, "group_size") == size_class) {
      max_value = ctx->max_block_size;
      default_value = ctx->cfg.default_block_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = ctx->max_grid_size;
      default_value = ctx->cfg.default_grid_size;
      // XXX: as a quick and dirty hack, use twice as many threads for
      // histograms by default.  We really should just be smarter
      // about sizes somehow.
      if (strstr(size_name, ".seghist_") != NULL) {
        default_value *= 2;
      }
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = ctx->max_tile_size;
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      // Threshold can be as large as it takes.
      default_value = ctx->cfg.default_threshold;
    } else {
      // Bespoke sizes have no limit or default.
    }

    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %zu (down from %zu)\n",
              size_name, max_value, *size_value);
      *size_value = max_value;
    }
  }
}

static void cuda_module_setup(struct cuda_context *ctx,
                              const char *src_fragments[],
                              const char *extra_opts[]) {
  char *ptx = NULL, *src = NULL;

  if (ctx->cfg.load_program_from == NULL) {
    src = concat_fragments(src_fragments);
  } else {
    src = slurp_file(ctx->cfg.load_program_from, NULL);
  }

  if (ctx->cfg.load_ptx_from) {
    if (ctx->cfg.load_program_from != NULL) {
      fprintf(stderr,
              "WARNING: Using PTX from %s instead of C code from %s\n",
              ctx->cfg.load_ptx_from, ctx->cfg.load_program_from);
    }
    ptx = slurp_file(ctx->cfg.load_ptx_from, NULL);
  }

  if (ctx->cfg.dump_program_to != NULL) {
    dump_file(ctx->cfg.dump_program_to, src, strlen(src));
  }

  if (ptx == NULL) {
    ptx = cuda_nvrtc_build(ctx, src, extra_opts);
  }

  if (ctx->cfg.dump_ptx_to != NULL) {
    dump_file(ctx->cfg.dump_ptx_to, ptx, strlen(ptx));
  }

  CUDA_SUCCEED(cuModuleLoadData(&ctx->module, ptx));

  free(ptx);
  if (src != NULL) {
    free(src);
  }
}

static void cuda_setup(struct cuda_context *ctx, const char *src_fragments[], const char *extra_opts[]) {
  CUDA_SUCCEED(cuInit(0));

  if (cuda_device_setup(ctx) != 0) {
    futhark_panic(-1, "No suitable CUDA device found.\n");
  }
  CUDA_SUCCEED(cuCtxCreate(&ctx->cu_ctx, 0, ctx->dev));

  free_list_init(&ctx->free_list);

  ctx->max_shared_memory = device_query(ctx->dev, MAX_SHARED_MEMORY_PER_BLOCK);
  ctx->max_block_size = device_query(ctx->dev, MAX_THREADS_PER_BLOCK);
  ctx->max_grid_size = device_query(ctx->dev, MAX_GRID_DIM_X);
  ctx->max_tile_size = sqrt(ctx->max_block_size);
  ctx->max_threshold = 0;
  ctx->max_bespoke = 0;
  ctx->lockstep_width = device_query(ctx->dev, WARP_SIZE);

  cuda_size_setup(ctx);
  cuda_module_setup(ctx, src_fragments, extra_opts);
}

// Count up the runtime all the profiling_records that occured during execution.
// Also clears the buffer of profiling_records.
static cudaError_t cuda_tally_profiling_records(struct cuda_context *ctx) {
  cudaError_t err;
  for (int i = 0; i < ctx->profiling_records_used; i++) {
    struct profiling_record record = ctx->profiling_records[i];

    float ms;
    if ((err = cudaEventElapsedTime(&ms, record.events[0], record.events[1])) != cudaSuccess) {
      return err;
    }

    // CUDA provides milisecond resolution, but we want microseconds.
    *record.runs += 1;
    *record.runtime += ms*1000;

    if ((err = cudaEventDestroy(record.events[0])) != cudaSuccess) {
      return err;
    }
    if ((err = cudaEventDestroy(record.events[1])) != cudaSuccess) {
      return err;
    }

    free(record.events);
  }

  ctx->profiling_records_used = 0;

  return cudaSuccess;
}

// Returns pointer to two events.
static cudaEvent_t* cuda_get_events(struct cuda_context *ctx, int *runs, int64_t *runtime) {
    if (ctx->profiling_records_used == ctx->profiling_records_capacity) {
      ctx->profiling_records_capacity *= 2;
      ctx->profiling_records =
        realloc(ctx->profiling_records,
                ctx->profiling_records_capacity *
                sizeof(struct profiling_record));
    }
    cudaEvent_t *events = calloc(2, sizeof(cudaEvent_t));
    cudaEventCreate(&events[0]);
    cudaEventCreate(&events[1]);
    ctx->profiling_records[ctx->profiling_records_used].events = events;
    ctx->profiling_records[ctx->profiling_records_used].runs = runs;
    ctx->profiling_records[ctx->profiling_records_used].runtime = runtime;
    ctx->profiling_records_used++;
    return events;
}

static CUresult cuda_free_all(struct cuda_context *ctx);

static void cuda_cleanup(struct cuda_context *ctx) {
  CUDA_SUCCEED(cuda_free_all(ctx));
  (void)cuda_tally_profiling_records(ctx);
  free(ctx->profiling_records);
  CUDA_SUCCEED(cuModuleUnload(ctx->module));
  CUDA_SUCCEED(cuCtxDestroy(ctx->cu_ctx));
}

static CUresult cuda_alloc(struct cuda_context *ctx, size_t min_size,
                           const char *tag, CUdeviceptr *mem_out) {
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;
  if (free_list_find(&ctx->free_list, min_size, &size, mem_out) == 0) {
    if (size >= min_size) {
      return CUDA_SUCCESS;
    } else {
      CUresult res = cuMemFree(*mem_out);
      if (res != CUDA_SUCCESS) {
        return res;
      }
    }
  }

  CUresult res = cuMemAlloc(mem_out, min_size);
  while (res == CUDA_ERROR_OUT_OF_MEMORY) {
    CUdeviceptr mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      res = cuMemFree(mem);
      if (res != CUDA_SUCCESS) {
        return res;
      }
    } else {
      break;
    }
    res = cuMemAlloc(mem_out, min_size);
  }

  return res;
}

static CUresult cuda_free(struct cuda_context *ctx, CUdeviceptr mem,
                          const char *tag) {
  size_t size;
  CUdeviceptr existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, -1, &size, &existing_mem) == 0) {
    CUresult res = cuMemFree(existing_mem);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }

  CUresult res = cuMemGetAddressRange(NULL, &size, mem);
  if (res == CUDA_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return res;
}

static CUresult cuda_free_all(struct cuda_context *ctx) {
  CUdeviceptr mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    CUresult res = cuMemFree(mem);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }

  return CUDA_SUCCESS;
}

// End of cuda.h.

const char *cuda_program[] =
           {"#define FUTHARK_CUDA\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long long int64_t;\ntypedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\ntypedef uint8_t uchar;\ntypedef uint16_t ushort;\ntypedef uint32_t uint;\ntypedef uint64_t ulong;\n#define __kernel extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK)\n#define __global\n#define __local\n#define __private\n#define __constant\n#define __write_only\n#define __read_only\nstatic inline int get_group_id_fn(int block_dim0, int block_dim1,\n                                  int block_dim2, int d)\n{\n    switch (d) {\n        \n      case 0:\n        d = block_dim0;\n        break;\n        \n      case 1:\n        d = block_dim1;\n        break;\n        \n      case 2:\n        d = block_dim2;\n        break;\n    }\n    switch (d) {\n        \n      case 0:\n        return blockIdx.x;\n        \n      case 1:\n        return blockIdx.y;\n        \n      case 2:\n        return blockIdx.z;\n        \n      default:\n        return 0;\n    }\n}\n#define get_group_id(d) get_group_id_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_num_groups_fn(int block_dim0, int block_dim1,\n                                    int block_dim2, int d)\n{\n    switch (d) {\n        \n      case 0:\n        d = block_dim0;\n        break;\n        \n      case 1:\n        d = block_dim1;\n        break;\n        \n      case 2:\n        d = block_dim2;\n        break;\n    }\n    switch (d) {\n        \n      case 0:\n        return gridDim.x;\n        \n      case 1:\n        return gridDim.y;\n        \n      case 2:\n        return gridDim.z;\n        \n      default:\n        return 0;\n    }\n}\n#define get_num_groups(d) get_num_groups_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_local_id(int d)\n{\n    switch (d) {\n        \n      case 0:\n        return threadIdx.x;\n        \n      case 1:\n        return threadIdx.y;\n        \n      case 2:\n        return threadI",
            "dx.z;\n        \n      default:\n        return 0;\n    }\n}\nstatic inline int get_local_size(int d)\n{\n    switch (d) {\n        \n      case 0:\n        return blockDim.x;\n        \n      case 1:\n        return blockDim.y;\n        \n      case 2:\n        return blockDim.z;\n        \n      default:\n        return 0;\n    }\n}\nstatic inline int get_global_id_fn(int block_dim0, int block_dim1,\n                                   int block_dim2, int d)\n{\n    return get_group_id(d) * get_local_size(d) + get_local_id(d);\n}\n#define get_global_id(d) get_global_id_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_global_size(int block_dim0, int block_dim1,\n                                  int block_dim2, int d)\n{\n    return get_num_groups(d) * get_local_size(d);\n}\n#define CLK_LOCAL_MEM_FENCE 1\n#define CLK_GLOBAL_MEM_FENCE 2\nstatic inline void barrier(int x)\n{\n    __syncthreads();\n}\nstatic inline void mem_fence_local()\n{\n    __threadfence_block();\n}\nstatic inline void mem_fence_global()\n{\n    __threadfence();\n}\n#define NAN (0.0/0.0)\n#define INFINITY (1.0/0.0)\nextern volatile __shared__ char shared_mem[];\nstatic inline uint8_t add8(uint8_t x, uint8_t y)\n{\n    return x + y;\n}\nstatic inline uint16_t add16(uint16_t x, uint16_t y)\n{\n    return x + y;\n}\nstatic inline uint32_t add32(uint32_t x, uint32_t y)\n{\n    return x + y;\n}\nstatic inline uint64_t add64(uint64_t x, uint64_t y)\n{\n    return x + y;\n}\nstatic inline uint8_t sub8(uint8_t x, uint8_t y)\n{\n    return x - y;\n}\nstatic inline uint16_t sub16(uint16_t x, uint16_t y)\n{\n    return x - y;\n}\nstatic inline uint32_t sub32(uint32_t x, uint32_t y)\n{\n    return x - y;\n}\nstatic inline uint64_t sub64(uint64_t x, uint64_t y)\n{\n    return x - y;\n}\nstatic inline uint8_t mul8(uint8_t x, uint8_t y)\n{\n    return x * y;\n}\nstatic inline uint16_t mul16(uint16_t x, uint16_t y)\n{\n    return x * y;\n}\nstatic inline uint32_t mul32(uint32_t x, uint32_t y)\n{\n    return x * y;\n}\nstatic inline uint64_t mul64(uint64_t x, uint64_t y)\n{\n    return x * y;\n",
            "}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t udiv_up8(uint8_t x, uint8_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint16_t udiv_up16(uint16_t x, uint16_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint32_t udiv_up32(uint32_t x, uint32_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint64_t udiv_up64(uint64_t x, uint64_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline uint8_t udiv_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint16_t udiv_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint32_t udiv_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint64_t udiv_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint8_t umod_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint16_t umod_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint32_t umod_safe32(uint32_t x, uint32_t y)\n{\n    r",
            "eturn y == 0 ? 0 : x % y;\n}\nstatic inline uint64_t umod_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t sdiv_up8(int8_t x, int8_t y)\n{\n    return sdiv8(x + y - 1, y);\n}\nstatic inline int16_t sdiv_up16(int16_t x, int16_t y)\n{\n    return sdiv16(x + y - 1, y);\n}\nstatic inline int32_t sdiv_up32(int32_t x, int32_t y)\n{\n    return sdiv32(x + y - 1, y);\n}\nstatic inline int64_t sdiv_up64(int64_t x, int64_t y)\n{\n    return sdiv64(x + y - 1, y);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t sdiv_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : sdiv8(x, y);\n}\nstatic inline int16_t sdiv_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : sdiv16(x, y);\n}\nstatic inline int32_t sdiv_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0",
            " : sdiv32(x, y);\n}\nstatic inline int64_t sdiv_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : sdiv64(x, y);\n}\nstatic inline int8_t sdiv_up_safe8(int8_t x, int8_t y)\n{\n    return sdiv_safe8(x + y - 1, y);\n}\nstatic inline int16_t sdiv_up_safe16(int16_t x, int16_t y)\n{\n    return sdiv_safe16(x + y - 1, y);\n}\nstatic inline int32_t sdiv_up_safe32(int32_t x, int32_t y)\n{\n    return sdiv_safe32(x + y - 1, y);\n}\nstatic inline int64_t sdiv_up_safe64(int64_t x, int64_t y)\n{\n    return sdiv_safe64(x + y - 1, y);\n}\nstatic inline int8_t smod_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : smod8(x, y);\n}\nstatic inline int16_t smod_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : smod16(x, y);\n}\nstatic inline int32_t smod_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : smod32(x, y);\n}\nstatic inline int64_t smod_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : smod64(x, y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t squot_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int16_t squot_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int32_t squot_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int64_t squot_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int8_t srem_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int16_t srem_safe16(int16_t x, int16_t y)\n{\n    retu",
            "rn y == 0 ? 0 : x % y;\n}\nstatic inline int32_t srem_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int64_t srem_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t ",
            "x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline bool ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline bool ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline bool ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline bool ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline bool ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline bool slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline bool slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic i",
            "nline bool slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline bool slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline bool sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\n#define sext_i8_i8(x) ((int8_t) (int8_t) x)\n#define sext_i8_i16(x) ((int16_t) (int8_t) x)\n#define sext_i8_i32(x) ((int32_t) (int8_t) x)\n#define sext_i8_i64(x) ((int64_t) (int8_t) x)\n#define sext_i16_i8(x) ((int8_t) (int16_t) x)\n#define sext_i16_i16(x) ((int1",
            "6_t) (int16_t) x)\n#define sext_i16_i32(x) ((int32_t) (int16_t) x)\n#define sext_i16_i64(x) ((int64_t) (int16_t) x)\n#define sext_i32_i8(x) ((int8_t) (int32_t) x)\n#define sext_i32_i16(x) ((int16_t) (int32_t) x)\n#define sext_i32_i32(x) ((int32_t) (int32_t) x)\n#define sext_i32_i64(x) ((int64_t) (int32_t) x)\n#define sext_i64_i8(x) ((int8_t) (int64_t) x)\n#define sext_i64_i16(x) ((int16_t) (int64_t) x)\n#define sext_i64_i32(x) ((int32_t) (int64_t) x)\n#define sext_i64_i64(x) ((int64_t) (int64_t) x)\n#define zext_i8_i8(x) ((int8_t) (uint8_t) x)\n#define zext_i8_i16(x) ((int16_t) (uint8_t) x)\n#define zext_i8_i32(x) ((int32_t) (uint8_t) x)\n#define zext_i8_i64(x) ((int64_t) (uint8_t) x)\n#define zext_i16_i8(x) ((int8_t) (uint16_t) x)\n#define zext_i16_i16(x) ((int16_t) (uint16_t) x)\n#define zext_i16_i32(x) ((int32_t) (uint16_t) x)\n#define zext_i16_i64(x) ((int64_t) (uint16_t) x)\n#define zext_i32_i8(x) ((int8_t) (uint32_t) x)\n#define zext_i32_i16(x) ((int16_t) (uint32_t) x)\n#define zext_i32_i32(x) ((int32_t) (uint32_t) x)\n#define zext_i32_i64(x) ((int64_t) (uint32_t) x)\n#define zext_i64_i8(x) ((int8_t) (uint64_t) x)\n#define zext_i64_i16(x) ((int16_t) (uint64_t) x)\n#define zext_i64_i32(x) ((int32_t) (uint64_t) x)\n#define zext_i64_i64(x) ((int64_t) (uint64_t) x)\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return popcount(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return __popc(zext_i8_i32(x));\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return __popc(zext_i16_i32(x));\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return __popc(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return __popcll(x);\n}\n#else\nstatic int32_t futrts_popc8(int8_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return ",
            "c;\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul_hi(a, b);\n}\n#elif defined(__CUDA_ARCH__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mulhi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul64hi(a, b);\n}\n#else\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    uint64_t aa = a;\n    uint64_t bb = b;\n    \n    return aa * bb >> 32;\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    __uint128_t aa = a;\n    __uint128_t bb = b;\n    \n    return aa * bb >> 64;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint32_t futrts_mad_hi32(uint",
            "32_t a, uint32_t b, uint32_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return mad_hi(a, b, c);\n}\n#else\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return futrts_mul_hi8(a, b) + c;\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return futrts_mul_hi16(a, b) + c;\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return futrts_mul_hi32(a, b) + c;\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return futrts_mul_hi64(a, b) + c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return clz(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return __clz(zext_i8_i32(x)) - 24;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return __clz(zext_i16_i32(x)) - 16;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return __clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return __clzll(x);\n}\n#else\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits",
            "; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    int i = 0;\n    \n    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    int i = 0;\n    \n    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    int i = 0;\n    \n    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    int i = 0;\n    \n    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 8 : y - 1;\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 16 : y - 1;\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 32 : y - 1;\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    int y = __ffsll(x);\n    \n    return y == 0 ? 64 : y - 1;\n}\n#else\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    return x == 0 ? 32 : __builtin_ctz(x);\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    return x == 0 ? 64 : __builtin_ctzll(x);\n}\n#endif\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return fmin(x, y);\n}\nstatic inline float fmax32(float x, float y)\n{\n    return fmax(x, y);\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inlin",
            "e bool cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline bool cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return (float) x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return (uint64_t) x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n",
            "{\n    return atan(x);\n}\nstatic inline float futrts_cosh32(float x)\n{\n    return cosh(x);\n}\nstatic inline float futrts_sinh32(float x)\n{\n    return sinh(x);\n}\nstatic inline float futrts_tanh32(float x)\n{\n    return tanh(x);\n}\nstatic inline float futrts_acosh32(float x)\n{\n    return acosh(x);\n}\nstatic inline float futrts_asinh32(float x)\n{\n    return asinh(x);\n}\nstatic inline float futrts_atanh32(float x)\n{\n    return atanh(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline bool futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    return fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return mad(a, b, c);\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fma(a, b, c);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t",
            ")\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return a * b + c;\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fmaf(a, b, c);\n}\n#endif\nstatic inline double fdiv64(double x, double y)\n{\n    return x / y;\n}\nstatic inline double fadd64(double x, double y)\n{\n    return x + y;\n}\nstatic inline double fsub64(double x, double y)\n{\n    return x - y;\n}\nstatic inline double fmul64(double x, double y)\n{\n    return x * y;\n}\nstatic inline double fmin64(double x, double y)\n{\n    return fmin(x, y);\n}\nstatic inline double fmax64(double x, double y)\n{\n    return fmax(x, y);\n}\nstatic inline double fpow64(double x, double y)\n{\n    return pow(x, y);\n}\nstatic inline bool cmplt64(double x, double y)\n{\n    return x < y;\n}\nstatic inline bool cmple64(double x, double y)\n{\n    return x <= y;\n}\nstatic inline double sitofp_i8_f64(int8_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i16_f64(int16_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i32_f64(int32_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i64_f64(int64_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i8_f64(uint8_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i16_f64(uint16_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i32_f64(uint32_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i64_f64(uint64_t x)\n{\n    return (double) x;\n}\nstatic inline int8_t fptosi_f64_i8(double x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f64_i16(double x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f64_i32(double x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f64_i64(double x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f64_i8(double x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f64_i16(double x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint32_t fptoui_f64_i32(double x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64",
            "_t fptoui_f64_i64(double x)\n{\n    return (uint64_t) x;\n}\nstatic inline double futrts_log64(double x)\n{\n    return log(x);\n}\nstatic inline double futrts_log2_64(double x)\n{\n    return log2(x);\n}\nstatic inline double futrts_log10_64(double x)\n{\n    return log10(x);\n}\nstatic inline double futrts_sqrt64(double x)\n{\n    return sqrt(x);\n}\nstatic inline double futrts_exp64(double x)\n{\n    return exp(x);\n}\nstatic inline double futrts_cos64(double x)\n{\n    return cos(x);\n}\nstatic inline double futrts_sin64(double x)\n{\n    return sin(x);\n}\nstatic inline double futrts_tan64(double x)\n{\n    return tan(x);\n}\nstatic inline double futrts_acos64(double x)\n{\n    return acos(x);\n}\nstatic inline double futrts_asin64(double x)\n{\n    return asin(x);\n}\nstatic inline double futrts_atan64(double x)\n{\n    return atan(x);\n}\nstatic inline double futrts_cosh64(double x)\n{\n    return cosh(x);\n}\nstatic inline double futrts_sinh64(double x)\n{\n    return sinh(x);\n}\nstatic inline double futrts_tanh64(double x)\n{\n    return tanh(x);\n}\nstatic inline double futrts_acosh64(double x)\n{\n    return acosh(x);\n}\nstatic inline double futrts_asinh64(double x)\n{\n    return asinh(x);\n}\nstatic inline double futrts_atanh64(double x)\n{\n    return atanh(x);\n}\nstatic inline double futrts_atan2_64(double x, double y)\n{\n    return atan2(x, y);\n}\nstatic inline double futrts_gamma64(double x)\n{\n    return tgamma(x);\n}\nstatic inline double futrts_lgamma64(double x)\n{\n    return lgamma(x);\n}\nstatic inline double futrts_fma64(double a, double b, double c)\n{\n    return fma(a, b, c);\n}\nstatic inline double futrts_round64(double x)\n{\n    return rint(x);\n}\nstatic inline double futrts_ceil64(double x)\n{\n    return ceil(x);\n}\nstatic inline double futrts_floor64(double x)\n{\n    return floor(x);\n}\nstatic inline bool futrts_isnan64(double x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf64(double x)\n{\n    return isinf(x);\n}\nstatic inline int64_t futrts_to_bits64(double x)\n{\n    union {\n        double f;\n        int64_t t;",
            "\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double futrts_from_bits64(int64_t x)\n{\n    union {\n        int64_t f;\n        double t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double fmod64(double x, double y)\n{\n    return fmod(x, y);\n}\n#ifdef __OPENCL_VERSION__\nstatic inline double futrts_lerp64(double v0, double v1, double t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline double futrts_mad64(double a, double b, double c)\n{\n    return mad(a, b, c);\n}\n#else\nstatic inline double futrts_lerp64(double v0, double v1, double t)\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline double futrts_mad64(double a, double b, double c)\n{\n    return a * b + c;\n}\n#endif\nstatic inline float fpconv_f32_f32(float x)\n{\n    return (float) x;\n}\nstatic inline double fpconv_f32_f64(float x)\n{\n    return (double) x;\n}\nstatic inline float fpconv_f64_f32(double x)\n{\n    return (float) x;\n}\nstatic inline double fpconv_f64_f64(double x)\n{\n    return (double) x;\n}\n// Start of atomics.h\n\ninline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline float atomic_fadd_f32_global(volatile __global float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline float atomic_fadd_f32_local(volatile __local float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do ",
            "{\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t",
            " atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\n// End of atomics.h\n\n\n\n\n__kernel void builtinzhreplicate_boolzireplicate_5007(int32_t num_elems_5004,\n                                                      unsigned char val_5005,\n                                                      __global\n                                                      unsigned char *mem_5003)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_",
            "gtid_5007;\n    int32_t replicate_ltid_5008;\n    int32_t replicate_gid_5009;\n    \n    replicate_gtid_5007 = get_global_id(0);\n    replicate_ltid_5008 = get_local_id(0);\n    replicate_gid_5009 = get_group_id(0);\n    if (slt64(replicate_gtid_5007, num_elems_5004)) {\n        ((__global bool *) mem_5003)[sext_i32_i64(replicate_gtid_5007)] =\n            val_5005;\n    }\n    \n  error_0:\n    return;\n}\n__kernel void builtinzhreplicate_i8zireplicate_4927(int32_t num_elems_4924,\n                                                    int8_t val_4925, __global\n                                                    unsigned char *mem_4923)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_4927;\n    int32_t replicate_ltid_4928;\n    int32_t replicate_gid_4929;\n    \n    replicate_gtid_4927 = get_global_id(0);\n    replicate_ltid_4928 = get_local_id(0);\n    replicate_gid_4929 = get_group_id(0);\n    if (slt64(replicate_gtid_4927, num_elems_4924)) {\n        ((__global int8_t *) mem_4923)[sext_i32_i64(replicate_gtid_4927)] =\n            val_4925;\n    }\n    \n  error_0:\n    return;\n}\n__kernel void mainzisegmap_4822(__global int *global_failure, int64_t n_4772,\n                                int64_t m_4773, __global\n                                unsigned char *mem_4895, __global\n                                unsigned char *mem_4897)\n{\n    #define segmap_group_sizze_4825 (mainzisegmap_group_sizze_4824)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_5012;\n    int32_t local_tid_5013;\n    int64_t group_sizze_5016;\n    int32_t wave_sizze_5015;\n    int32_t group_tid_5014;\n    \n    global_tid_5012 = get_global_id(0);\n    local_tid_5013 = get_local_id(0);\n    group_sizze_5016 = get_local_size(0);\n    wave_sizze_5015 = LOCKSTEP_WIDTH;\n    group_tid_5014 = get_group_id(0);\n    \n    int32_t phys_tid_4822;\n    \n",
            "    phys_tid_4822 = global_tid_5012;\n    \n    int64_t write_i_4821;\n    \n    write_i_4821 = sext_i32_i64(group_tid_5014) * segmap_group_sizze_4825 +\n        sext_i32_i64(local_tid_5013);\n    if (slt64(write_i_4821, m_4773)) {\n        int64_t write_index_4786 = ((__global\n                                     int64_t *) mem_4895)[write_i_4821];\n        \n        if (sle64((int64_t) 0, write_index_4786) && slt64(write_index_4786,\n                                                          n_4772)) {\n            ((__global bool *) mem_4897)[write_index_4786] = 1;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_4825\n}\n__kernel void mainzisegmap_4862(__global int *global_failure,\n                                int failure_is_an_option, __global\n                                int64_t *global_failure_args, int64_t n_4772,\n                                int64_t m_4773, __global\n                                unsigned char *mem_4895, __global\n                                unsigned char *mem_4903, __global\n                                unsigned char *mem_4906)\n{\n    #define segmap_group_sizze_4875 (mainzisegmap_group_sizze_4864)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_5150;\n    int32_t local_tid_5151;\n    int64_t group_sizze_5154;\n    int32_t wave_sizze_5153;\n    int32_t group_tid_5152;\n    \n    global_tid_5150 = get_global_id(0);\n    local_tid_5151 = get_local_id(0);\n    group_sizze_5154 = get_local_size(0);\n    wave_sizze_5153 = LOCKSTEP_WIDTH;\n    group_tid_5152 = get_group_id(0);\n    \n    int32_t phys_tid_4862;\n    \n    phys_tid_4862 = global_tid_5150;\n    \n    int64_t gtid_4861;\n    \n    gtid_4861 = sext_i32_i64(group_tid_5152) * segmap_group_sizze_4875 +\n        sext_i32_i64(local_tid_5151);\n    if (slt64(gtid_4861, m_4773)) {\n        int64_t x_4878 = ((__global int64_t *) mem_4895)[gtid_4861];\n        int64_",
            "t defunc_0_f_res_4879 = sub64(x_4878, (int64_t) 1);\n        bool x_4880 = sle64((int64_t) 0, defunc_0_f_res_4879);\n        bool y_4881 = slt64(defunc_0_f_res_4879, n_4772);\n        bool bounds_check_4882 = x_4880 && y_4881;\n        bool index_certs_4883;\n        \n        if (!bounds_check_4882) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 1) == -1) {\n                    global_failure_args[0] = defunc_0_f_res_4879;\n                    global_failure_args[1] = n_4772;\n                    ;\n                }\n                return;\n            }\n        }\n        \n        float break_arg_4884 = ((__global\n                                 float *) mem_4903)[defunc_0_f_res_4879];\n        \n        ((__global float *) mem_4906)[gtid_4861] = break_arg_4884;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_4875\n}\n__kernel void mainzisegscan_4820(__global int *global_failure,\n                                 uint local_mem_4941_backing_offset_0,\n                                 int64_t m_4773, __global\n                                 unsigned char *shp_mem_4891, __global\n                                 unsigned char *mem_4895, __global\n                                 unsigned char *mainziid_counter_mem_4915,\n                                 __global unsigned char *status_flags_mem_4917,\n                                 __global unsigned char *aggregates_mem_4919,\n                                 __global unsigned char *incprefixes_mem_4921)\n{\n    #define segscan_group_sizze_4815 (mainzisegscan_group_sizze_4814)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *local_mem_4941_backing_0 =\n                  &shared_mem[local_mem_4941_backing_offset_0];\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_4932;\n    int32_t local_tid_4933;\n    int64_t group_sizze_4936;\n    int32_t wave_sizze_4935;\n    int32_t group_tid_4934;\n    \n    glo",
            "bal_tid_4932 = get_global_id(0);\n    local_tid_4933 = get_local_id(0);\n    group_sizze_4936 = get_local_size(0);\n    wave_sizze_4935 = LOCKSTEP_WIDTH;\n    group_tid_4934 = get_group_id(0);\n    \n    int32_t phys_tid_4820;\n    \n    phys_tid_4820 = global_tid_4932;\n    \n    int64_t byte_offsets_4937;\n    \n    byte_offsets_4937 = (int64_t) 0;\n    \n    int64_t byte_offsets_4938;\n    \n    byte_offsets_4938 = segscan_group_sizze_4815 * (int64_t) 8;\n    \n    int64_t warp_byte_offset_4939;\n    \n    warp_byte_offset_4939 = (int64_t) 32;\n    \n    int64_t warp_byte_offset_4940;\n    \n    warp_byte_offset_4940 = (int64_t) 288;\n    // Allocate reused shared memeory\n    { }\n    \n    __local char *local_mem_4941;\n    \n    local_mem_4941 = (__local char *) local_mem_4941_backing_0;\n    \n    int64_t trans_arr_len_4942;\n    \n    trans_arr_len_4942 = (int64_t) 9 * segscan_group_sizze_4815;\n    \n    int32_t dynamic_id_4949;\n    \n    if (local_tid_4933 == 0) {\n        dynamic_id_4949 = atomic_add_i32_global(&((volatile __global\n                                                   int *) mainziid_counter_mem_4915)[(int64_t) 0],\n                                                (int) 1);\n        ((__local int32_t *) local_mem_4941)[(int64_t) 0] = dynamic_id_4949;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    dynamic_id_4949 = ((__local int32_t *) local_mem_4941)[(int64_t) 0];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int64_t blockOff_4950;\n    \n    blockOff_4950 = sext_i32_i64(dynamic_id_4949) * (int64_t) 9 *\n        segscan_group_sizze_4815;\n    \n    int64_t private_mem_4951[9];\n    \n    // Load and map\n    {\n        for (int64_t i_4953 = 0; i_4953 < (int64_t) 9; i_4953++) {\n            int64_t gtid_4819 = blockOff_4950 + sext_i32_i64(local_tid_4933) +\n                    i_4953 * segscan_group_sizze_4815;\n            \n            if (slt64(gtid_4819, m_4773)) {\n                int64_t x_4783 = ((__global int64_t *) shp_mem_4891)[gtid_4819];\n                \n                private_mem_4951[i_4",
            "953] = x_4783;\n            } else {\n                private_mem_4951[i_4953] = (int64_t) 0;\n            }\n        }\n    }\n    // Transpose scan inputs\n    {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_4954 = 0; i_4954 < (int64_t) 9; i_4954++) {\n            int64_t sharedIdx_4955 = sext_i32_i64(local_tid_4933) + i_4954 *\n                    segscan_group_sizze_4815;\n            \n            ((__local int64_t *) local_mem_4941)[sharedIdx_4955] =\n                private_mem_4951[i_4954];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t i_4956 = 0; i_4956 < 9; i_4956++) {\n            int32_t sharedIdx_4957 = local_tid_4933 * 9 + i_4956;\n            \n            private_mem_4951[sext_i32_i64(i_4956)] = ((__local\n                                                       int64_t *) local_mem_4941)[sext_i32_i64(sharedIdx_4957)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    // Per thread scan\n    {\n        for (int64_t i_4958 = 0; i_4958 < (int64_t) 8; i_4958++) {\n            int64_t x_4780;\n            int64_t x_4781;\n            \n            x_4780 = private_mem_4951[i_4958];\n            x_4781 = private_mem_4951[i_4958 + (int64_t) 1];\n            \n            int64_t defunc_1_op_res_4782;\n            \n            defunc_1_op_res_4782 = add64(x_4780, x_4781);\n            private_mem_4951[i_4958 + (int64_t) 1] = defunc_1_op_res_4782;\n        }\n    }\n    // Publish results in shared memory\n    {\n        ((__local int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                     (int64_t) 8) +\n                                             sext_i32_i64(local_tid_4933)] =\n            private_mem_4951[(int64_t) 8];\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n    int64_t acc_4962;\n    \n    // Scan results (with warp scan)\n    {\n        int64_t x_4959;\n        int64_t x_4960;\n        int64_t x_4963;\n        int64_t x_4964;\n        bool ltid_in_bounds_4966;\n        \n        ltid_in_bounds_49",
            "66 = slt64(sext_i32_i64(local_tid_4933),\n                                    segscan_group_sizze_4815);\n        \n        int32_t skip_threads_4967;\n        \n        // read input for in-block scan\n        {\n            if (ltid_in_bounds_4966) {\n                x_4960 = ((volatile __local\n                           int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                              (int64_t) 8) +\n                                                      sext_i32_i64(local_tid_4933)];\n                if ((local_tid_4933 - squot32(local_tid_4933, 32) * 32) == 0) {\n                    x_4959 = x_4960;\n                }\n            }\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_4967 = 1;\n            while (slt32(skip_threads_4967, 32)) {\n                if (sle32(skip_threads_4967, local_tid_4933 -\n                          squot32(local_tid_4933, 32) * 32) &&\n                    ltid_in_bounds_4966) {\n                    // read operands\n                    {\n                        x_4959 = ((volatile __local\n                                   int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                                      (int64_t) 8) +\n                                                              (sext_i32_i64(local_tid_4933) -\n                                                               sext_i32_i64(skip_threads_4967))];\n                    }\n                    // perform operation\n                    {\n                        int64_t defunc_1_op_res_4961 = add64(x_4959, x_4960);\n                        \n                        x_4959 = defunc_1_op_res_4961;\n                    }\n                }\n                if (sle32(wave_sizze_4935, skip_threads_4967)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_4967, local_tid_4933 -\n                          squot32",
            "(local_tid_4933, 32) * 32) &&\n                    ltid_in_bounds_4966) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                             (int64_t) 8) +\n                                                     sext_i32_i64(local_tid_4933)] =\n                            x_4959;\n                        x_4960 = x_4959;\n                    }\n                }\n                if (sle32(wave_sizze_4935, skip_threads_4967)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_4967 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_4933 - squot32(local_tid_4933, 32) * 32) == 31 &&\n                ltid_in_bounds_4966) {\n                ((volatile __local\n                  int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                     (int64_t) 8) +\n                                             sext_i32_i64(squot32(local_tid_4933,\n                                                                  32))] =\n                    x_4959;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n        {\n            int32_t skip_threads_4968;\n            \n            // read input for in-block scan\n            {\n                if (squot32(local_tid_4933, 32) == 0 && ltid_in_bounds_4966) {\n                    x_4964 = ((volatile __local\n                               int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                                  (int64_t) 8) +\n                                                          sext_i32_i64(local_tid_4933)];\n                    if ((l",
            "ocal_tid_4933 - squot32(local_tid_4933, 32) * 32) ==\n                        0) {\n                        x_4963 = x_4964;\n                    }\n                }\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_4968 = 1;\n                while (slt32(skip_threads_4968, 32)) {\n                    if (sle32(skip_threads_4968, local_tid_4933 -\n                              squot32(local_tid_4933, 32) * 32) &&\n                        (squot32(local_tid_4933, 32) == 0 &&\n                         ltid_in_bounds_4966)) {\n                        // read operands\n                        {\n                            x_4963 = ((volatile __local\n                                       int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                                          (int64_t) 8) +\n                                                                  (sext_i32_i64(local_tid_4933) -\n                                                                   sext_i32_i64(skip_threads_4968))];\n                        }\n                        // perform operation\n                        {\n                            int64_t defunc_1_op_res_4965 = add64(x_4963,\n                                                                 x_4964);\n                            \n                            x_4963 = defunc_1_op_res_4965;\n                        }\n                    }\n                    if (sle32(wave_sizze_4935, skip_threads_4968)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_4968, local_tid_4933 -\n                              squot32(local_tid_4933, 32) * 32) &&\n                        (squot32(local_tid_4933, 32) == 0 &&\n                         ltid_in_bounds_4966)) {\n                        // write result\n                        {\n                            ((volatile __local\n                          ",
            "    int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                                 (int64_t) 8) +\n                                                         sext_i32_i64(local_tid_4933)] =\n                                x_4963;\n                            x_4964 = x_4963;\n                        }\n                    }\n                    if (sle32(wave_sizze_4935, skip_threads_4968)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_4968 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_4933, 32) == 0 || !ltid_in_bounds_4966)) {\n                // read operands\n                {\n                    x_4960 = x_4959;\n                    x_4959 = ((__local\n                               int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                                  (int64_t) 8) +\n                                                          (sext_i32_i64(squot32(local_tid_4933,\n                                                                                32)) -\n                                                           (int64_t) 1)];\n                }\n                // perform operation\n                {\n                    int64_t defunc_1_op_res_4961 = add64(x_4959, x_4960);\n                    \n                    x_4959 = defunc_1_op_res_4961;\n                }\n                // write final result\n                {\n                    ((__local\n                      int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                         (int64_t) 8) +\n                                                 sext_i32_i64(local_tid_4933)] =\n                        x_4959;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // rest",
            "ore correct values for first block\n        {\n            if (squot32(local_tid_4933, 32) == 0) {\n                ((__local int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                             (int64_t) 8) +\n                                                     sext_i32_i64(local_tid_4933)] =\n                    x_4960;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_4933 == 0) {\n            acc_4962 = ((__local\n                         int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                            (int64_t) 8) +\n                                                    (segscan_group_sizze_4815 -\n                                                     (int64_t) 1)];\n        } else {\n            acc_4962 = ((__local\n                         int64_t *) local_mem_4941)[squot64(byte_offsets_4937,\n                                                            (int64_t) 8) +\n                                                    (sext_i32_i64(local_tid_4933) -\n                                                     (int64_t) 1)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n    int64_t prefix_4969;\n    \n    prefix_4969 = (int64_t) 0;\n    // Perform lookback\n    {\n        if (dynamic_id_4949 == (int64_t) 0 && local_tid_4933 == 0) {\n            ((volatile __global\n              int64_t *) incprefixes_mem_4921)[dynamic_id_4949] = acc_4962;\n            mem_fence_global();\n            ((volatile __global\n              int8_t *) status_flags_mem_4917)[dynamic_id_4949] = (int8_t) 2;\n            acc_4962 = (int64_t) 0;\n        }\n        if (!(dynamic_id_4949 == (int64_t) 0) && slt32(local_tid_4933,\n                                                       wave_sizze_4935)) {\n            if (local_tid_4933 == 0) {\n                ((volatile __global\n                  int64_t *) aggregates_mem_4919)[dynamic_id_4949] = acc",
            "_4962;\n                mem_fence_global();\n                ((volatile __global\n                  int8_t *) status_flags_mem_4917)[dynamic_id_4949] =\n                    (int8_t) 1;\n                ((__local int8_t *) local_mem_4941)[(int64_t) 0] = ((__global\n                                                                     int8_t *) status_flags_mem_4917)[dynamic_id_4949 -\n                                                                                                      (int64_t) 1];\n            }\n            mem_fence_local();\n            \n            int8_t status_4970;\n            \n            status_4970 = ((__local int8_t *) local_mem_4941)[(int64_t) 0];\n            if (status_4970 == (int8_t) 2) {\n                if (local_tid_4933 == 0) {\n                    prefix_4969 = ((volatile __global\n                                    int64_t *) incprefixes_mem_4921)[dynamic_id_4949 -\n                                                                     (int64_t) 1];\n                }\n            } else {\n                int32_t readOffset_4971 = dynamic_id_4949 -\n                        sext_i32_i64(wave_sizze_4935);\n                \n                while (slt32(wave_sizze_4935 * -1, readOffset_4971)) {\n                    int32_t read_i_4972 = readOffset_4971 + local_tid_4933;\n                    int64_t aggr_4973 = (int64_t) 0;\n                    int8_t flag_4974 = (int8_t) 0;\n                    int8_t used_4975 = (int8_t) 0;\n                    \n                    if (sle32(0, read_i_4972)) {\n                        flag_4974 = ((volatile __global\n                                      int8_t *) status_flags_mem_4917)[sext_i32_i64(read_i_4972)];\n                        if (flag_4974 == (int8_t) 2) {\n                            aggr_4973 = ((volatile __global\n                                          int64_t *) incprefixes_mem_4921)[sext_i32_i64(read_i_4972)];\n                        } else {\n                            if (flag_4974 == (int8_t) 1) {\n      ",
            "                          aggr_4973 = ((volatile __global\n                                              int64_t *) aggregates_mem_4919)[sext_i32_i64(read_i_4972)];\n                                used_4975 = (int8_t) 1;\n                            }\n                        }\n                    }\n                    ((__local\n                      int64_t *) local_mem_4941)[squot64(warp_byte_offset_4939,\n                                                         (int64_t) 8) +\n                                                 sext_i32_i64(local_tid_4933)] =\n                        aggr_4973;\n                    \n                    int8_t tmp_4976;\n                    \n                    tmp_4976 = flag_4974 | used_4975 << (int8_t) 2;\n                    ((__local\n                      int8_t *) local_mem_4941)[sext_i32_i64(local_tid_4933)] =\n                        tmp_4976;\n                    mem_fence_local();\n                    flag_4974 = ((volatile __local\n                                  int8_t *) local_mem_4941)[sext_i32_i64(wave_sizze_4935) -\n                                                            (int64_t) 1];\n                    if (local_tid_4933 == 0) {\n                        if (!(flag_4974 == (int8_t) 2)) {\n                            int64_t x_4977 = (int64_t) 0;\n                            int64_t x_4978;\n                            int8_t flag1_4980;\n                            \n                            flag1_4980 = (int8_t) 0;\n                            \n                            int8_t flag2_4981;\n                            int8_t used1_4982;\n                            \n                            used1_4982 = (int8_t) 0;\n                            \n                            int8_t used2_4983;\n                            \n                            for (int32_t i_4984 = 0; i_4984 < wave_sizze_4935;\n                                 i_4984++) {\n                                flag2_4981 = ((__local\n                                ",
            "               int8_t *) local_mem_4941)[sext_i32_i64(i_4984)];\n                                used2_4983 = lshr8(flag2_4981, (int8_t) 2);\n                                flag2_4981 = flag2_4981 & (int8_t) 3;\n                                x_4978 = ((__local\n                                           int64_t *) local_mem_4941)[squot64(warp_byte_offset_4939,\n                                                                              (int64_t) 8) +\n                                                                      sext_i32_i64(i_4984)];\n                                if (!(flag2_4981 == (int8_t) 1)) {\n                                    flag1_4980 = flag2_4981;\n                                    used1_4982 = used2_4983;\n                                    x_4977 = x_4978;\n                                } else {\n                                    used1_4982 += used2_4983;\n                                    \n                                    int64_t defunc_1_op_res_4979;\n                                    \n                                    defunc_1_op_res_4979 = add64(x_4977,\n                                                                 x_4978);\n                                    x_4977 = defunc_1_op_res_4979;\n                                }\n                            }\n                            flag_4974 = flag1_4980;\n                            used_4975 = used1_4982;\n                            aggr_4973 = x_4977;\n                        } else {\n                            aggr_4973 = ((__local\n                                          int64_t *) local_mem_4941)[squot64(warp_byte_offset_4939,\n                                                                             (int64_t) 8) +\n                                                                     (sext_i32_i64(wave_sizze_4935) -\n                                                                      (int64_t) 1)];\n                        }\n                        if (flag_4974 == (int8_t)",
            " 2) {\n                            readOffset_4971 = wave_sizze_4935 * -1;\n                        } else {\n                            readOffset_4971 -= zext_i8_i32(used_4975);\n                        }\n                        ((__local int32_t *) local_mem_4941)[(int64_t) 0] =\n                            readOffset_4971;\n                        \n                        int64_t x_4985;\n                        \n                        x_4985 = aggr_4973;\n                        \n                        int64_t x_4986;\n                        \n                        x_4986 = prefix_4969;\n                        \n                        int64_t defunc_1_op_res_4987;\n                        \n                        defunc_1_op_res_4987 = add64(x_4985, x_4986);\n                        prefix_4969 = defunc_1_op_res_4987;\n                    }\n                    mem_fence_local();\n                    readOffset_4971 = ((__local\n                                        int32_t *) local_mem_4941)[(int64_t) 0];\n                }\n            }\n            if (local_tid_4933 == 0) {\n                int64_t x_4988 = prefix_4969;\n                int64_t x_4989 = acc_4962;\n                int64_t defunc_1_op_res_4990 = add64(x_4988, x_4989);\n                \n                ((volatile __global\n                  int64_t *) incprefixes_mem_4921)[dynamic_id_4949] =\n                    defunc_1_op_res_4990;\n                mem_fence_global();\n                ((volatile __global\n                  int8_t *) status_flags_mem_4917)[dynamic_id_4949] =\n                    (int8_t) 2;\n                ((__local\n                  int64_t *) local_mem_4941)[squot64(warp_byte_offset_4939,\n                                                     (int64_t) 8)] =\n                    prefix_4969;\n                acc_4962 = (int64_t) 0;\n            }\n        }\n        if (!(dynamic_id_4949 == (int64_t) 0)) {\n            barrier(CLK_LOCAL_MEM_FENCE);\n            prefix_4969 = ((__local\n                 ",
            "           int64_t *) local_mem_4941)[squot64(warp_byte_offset_4939,\n                                                               (int64_t) 8)];\n            barrier(CLK_LOCAL_MEM_FENCE);\n        }\n    }\n    // Distribute results\n    {\n        int64_t x_4991;\n        int64_t x_4992;\n        int64_t x_4994;\n        \n        x_4994 = prefix_4969;\n        \n        int64_t x_4995;\n        \n        x_4995 = acc_4962;\n        \n        int64_t defunc_1_op_res_4996;\n        \n        defunc_1_op_res_4996 = add64(x_4994, x_4995);\n        x_4991 = defunc_1_op_res_4996;\n        for (int64_t i_4997 = 0; i_4997 < (int64_t) 9; i_4997++) {\n            x_4992 = private_mem_4951[i_4997];\n            \n            int64_t defunc_1_op_res_4993;\n            \n            defunc_1_op_res_4993 = add64(x_4991, x_4992);\n            private_mem_4951[i_4997] = defunc_1_op_res_4993;\n        }\n    }\n    // Transpose scan output\n    {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_4998 = 0; i_4998 < (int64_t) 9; i_4998++) {\n            int64_t sharedIdx_4999 = sext_i32_i64(local_tid_4933 * 9) + i_4998;\n            \n            ((__local int64_t *) local_mem_4941)[sharedIdx_4999] =\n                private_mem_4951[i_4998];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5000 = 0; i_5000 < (int64_t) 9; i_5000++) {\n            int32_t sharedIdx_5001 = local_tid_4933 +\n                    sext_i64_i32(segscan_group_sizze_4815 * i_5000);\n            \n            private_mem_4951[i_5000] = ((__local\n                                         int64_t *) local_mem_4941)[sext_i32_i64(sharedIdx_5001)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    // Write block scan results to global memory\n    {\n        for (int64_t i_5002 = 0; i_5002 < (int64_t) 9; i_5002++) {\n            int64_t gtid_4819 = blockOff_4950 + segscan_group_sizze_4815 *\n                    i_5002 + sext_i32_i64(local_tid_4933);\n            \n            if (slt64(gtid_4819, m_4773)) {\n         ",
            "       ((__global int64_t *) mem_4895)[gtid_4819] =\n                    private_mem_4951[i_5002];\n            }\n        }\n    }\n    // If this is the last block, reset the dynamicId\n    {\n        if (dynamic_id_4949 == sdiv_up64(m_4773, segscan_group_sizze_4815 *\n                                         (int64_t) 9) - (int64_t) 1) {\n            ((__global int32_t *) mainziid_counter_mem_4915)[(int64_t) 0] = 0;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segscan_group_sizze_4815\n}\n__kernel void mainzisegscan_4834(__global int *global_failure,\n                                 int failure_is_an_option, __global\n                                 int64_t *global_failure_args,\n                                 uint local_mem_5043_backing_offset_0,\n                                 int64_t n_4772, int64_t implz2080U_4774,\n                                 __global unsigned char *mat_inds_mem_4889,\n                                 __global unsigned char *mat_vals_mem_4890,\n                                 __global unsigned char *vct_mem_4892, __global\n                                 unsigned char *mem_4897, __global\n                                 unsigned char *mem_4901, __global\n                                 unsigned char *mem_4903, __global\n                                 unsigned char *mainziid_counter_mem_5020,\n                                 __global unsigned char *status_flags_mem_5022,\n                                 __global unsigned char *aggregates_mem_5024,\n                                 __global unsigned char *incprefixes_mem_5026,\n                                 __global unsigned char *aggregates_mem_5028,\n                                 __global unsigned char *incprefixes_mem_5030)\n{\n    #define segscan_group_sizze_4829 (mainzisegscan_group_sizze_4828)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *local_mem_5043_backing_0 =\n                  &shared_mem[local_mem_5043_backi",
            "ng_offset_0];\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_5032;\n    int32_t local_tid_5033;\n    int64_t group_sizze_5036;\n    int32_t wave_sizze_5035;\n    int32_t group_tid_5034;\n    \n    global_tid_5032 = get_global_id(0);\n    local_tid_5033 = get_local_id(0);\n    group_sizze_5036 = get_local_size(0);\n    wave_sizze_5035 = LOCKSTEP_WIDTH;\n    group_tid_5034 = get_group_id(0);\n    \n    int32_t phys_tid_4834;\n    \n    phys_tid_4834 = global_tid_5032;\n    \n    int64_t byte_offsets_5037;\n    \n    byte_offsets_5037 = (int64_t) 0;\n    \n    int64_t byte_offsets_5038;\n    \n    byte_offsets_5038 = segscan_group_sizze_4829;\n    \n    int64_t byte_offsets_5039;\n    \n    byte_offsets_5039 = sdiv_up64(segscan_group_sizze_4829, (int64_t) 4) *\n        (int64_t) 4 + segscan_group_sizze_4829 * (int64_t) 4;\n    \n    int64_t warp_byte_offset_5040;\n    \n    warp_byte_offset_5040 = (int64_t) 32;\n    \n    int64_t warp_byte_offset_5041;\n    \n    warp_byte_offset_5041 = (int64_t) 64;\n    \n    int64_t warp_byte_offset_5042;\n    \n    warp_byte_offset_5042 = (int64_t) 192;\n    // Allocate reused shared memeory\n    { }\n    \n    __local char *local_mem_5043;\n    \n    local_mem_5043 = (__local char *) local_mem_5043_backing_0;\n    \n    int64_t trans_arr_len_5044;\n    \n    trans_arr_len_5044 = (int64_t) 9 * segscan_group_sizze_4829;\n    \n    int32_t dynamic_id_5054;\n    \n    if (local_tid_5033 == 0) {\n        dynamic_id_5054 = atomic_add_i32_global(&((volatile __global\n                                                   int *) mainziid_counter_mem_5020)[(int64_t) 0],\n                                                (int) 1);\n        ((__local int32_t *) local_mem_5043)[(int64_t) 0] = dynamic_id_5054;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    dynamic_id_5054 = ((__local int32_t *)",
            " local_mem_5043)[(int64_t) 0];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int64_t blockOff_5055;\n    \n    blockOff_5055 = sext_i32_i64(dynamic_id_5054) * (int64_t) 9 *\n        segscan_group_sizze_4829;\n    \n    bool private_mem_5056[9];\n    float private_mem_5058[9];\n    \n    // Load and map\n    {\n        for (int64_t i_5060 = 0; i_5060 < (int64_t) 9; i_5060++) {\n            int64_t gtid_4833 = blockOff_5055 + sext_i32_i64(local_tid_5033) +\n                    i_5060 * segscan_group_sizze_4829;\n            \n            if (slt64(gtid_4833, n_4772)) {\n                int64_t x_4796 = ((__global\n                                   int64_t *) mat_inds_mem_4889)[gtid_4833];\n                float x_4797 = ((__global\n                                 float *) mat_vals_mem_4890)[gtid_4833];\n                bool x_4798 = ((__global bool *) mem_4897)[gtid_4833];\n                bool x_4799 = sle64((int64_t) 0, x_4796);\n                bool y_4800 = slt64(x_4796, implz2080U_4774);\n                bool bounds_check_4801 = x_4799 && y_4800;\n                bool index_certs_4802;\n                \n                if (!bounds_check_4801) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 0) ==\n                            -1) {\n                            global_failure_args[0] = x_4796;\n                            global_failure_args[1] = implz2080U_4774;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                float y_4803 = ((__global float *) vct_mem_4892)[x_4796];\n                float defunc_0_f_res_4804 = x_4797 * y_4803;\n                \n                private_mem_5056[i_5060] = x_4798;\n                private_mem_5058[i_5060] = defunc_0_f_res_4804;\n            } else {\n                private_mem_5056[i_5060] = 0;\n                private_mem_5058[i_5060] = 0.0F;\n            ",
            "}\n        }\n    }\n    // Transpose scan inputs\n    {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5061 = 0; i_5061 < (int64_t) 9; i_5061++) {\n            int64_t sharedIdx_5062 = sext_i32_i64(local_tid_5033) + i_5061 *\n                    segscan_group_sizze_4829;\n            \n            ((__local bool *) local_mem_5043)[sharedIdx_5062] =\n                private_mem_5056[i_5061];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t i_5063 = 0; i_5063 < 9; i_5063++) {\n            int32_t sharedIdx_5064 = local_tid_5033 * 9 + i_5063;\n            \n            private_mem_5056[sext_i32_i64(i_5063)] = ((__local\n                                                       bool *) local_mem_5043)[sext_i32_i64(sharedIdx_5064)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5065 = 0; i_5065 < (int64_t) 9; i_5065++) {\n            int64_t sharedIdx_5066 = sext_i32_i64(local_tid_5033) + i_5065 *\n                    segscan_group_sizze_4829;\n            \n            ((__local float *) local_mem_5043)[sharedIdx_5066] =\n                private_mem_5058[i_5065];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t i_5067 = 0; i_5067 < 9; i_5067++) {\n            int32_t sharedIdx_5068 = local_tid_5033 * 9 + i_5067;\n            \n            private_mem_5058[sext_i32_i64(i_5067)] = ((__local\n                                                       float *) local_mem_5043)[sext_i32_i64(sharedIdx_5068)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    // Per thread scan\n    {\n        for (int64_t i_5069 = 0; i_5069 < (int64_t) 8; i_5069++) {\n            bool x_4789;\n            bool x_4791;\n            \n            x_4789 = private_mem_5056[i_5069];\n            x_4791 = private_mem_5056[i_5069 + (int64_t) 1];\n            \n            float x_4790;\n            float x_4792;\n            \n            x_4790 = private_mem_5058[i_5069];\n            x_4792 = private_mem_5058[i_5069 + (int64_t) 1];\n            \n       ",
            "     bool f_4793;\n            \n            f_4793 = x_4789 || x_4791;\n            \n            float defunc_1_op_res_4794;\n            \n            if (x_4791) {\n                defunc_1_op_res_4794 = x_4792;\n            } else {\n                float defunc_1_op_res_f_res_4795 = x_4790 + x_4792;\n                \n                defunc_1_op_res_4794 = defunc_1_op_res_f_res_4795;\n            }\n            private_mem_5056[i_5069 + (int64_t) 1] = f_4793;\n            private_mem_5058[i_5069 + (int64_t) 1] = defunc_1_op_res_4794;\n        }\n    }\n    // Publish results in shared memory\n    {\n        ((__local bool *) local_mem_5043)[byte_offsets_5037 +\n                                          sext_i32_i64(local_tid_5033)] =\n            private_mem_5056[(int64_t) 8];\n        ((__local float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                   (int64_t) 4) +\n                                           sext_i32_i64(local_tid_5033)] =\n            private_mem_5058[(int64_t) 8];\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n    bool acc_5077;\n    float acc_5078;\n    \n    // Scan results (with warp scan)\n    {\n        bool x_5070;\n        float x_5071;\n        bool x_5072;\n        float x_5073;\n        bool x_5079;\n        float x_5080;\n        bool x_5081;\n        float x_5082;\n        bool ltid_in_bounds_5086;\n        \n        ltid_in_bounds_5086 = slt64(sext_i32_i64(local_tid_5033),\n                                    segscan_group_sizze_4829);\n        \n        int32_t skip_threads_5087;\n        \n        // read input for in-block scan\n        {\n            if (ltid_in_bounds_5086) {\n                x_5072 = ((volatile __local\n                           bool *) local_mem_5043)[byte_offsets_5037 +\n                                                   sext_i32_i64(local_tid_5033)];\n                x_5073 = ((volatile __local\n                           float *) local_mem_5043)[squot64(byte_offsets_5038,\n                               ",
            "                             (int64_t) 4) +\n                                                    sext_i32_i64(local_tid_5033)];\n                if ((local_tid_5033 - squot32(local_tid_5033, 32) * 32) == 0) {\n                    x_5070 = x_5072;\n                    x_5071 = x_5073;\n                }\n            }\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_5087 = 1;\n            while (slt32(skip_threads_5087, 32)) {\n                if (sle32(skip_threads_5087, local_tid_5033 -\n                          squot32(local_tid_5033, 32) * 32) &&\n                    ltid_in_bounds_5086) {\n                    // read operands\n                    {\n                        x_5070 = ((volatile __local\n                                   bool *) local_mem_5043)[byte_offsets_5037 +\n                                                           (sext_i32_i64(local_tid_5033) -\n                                                            sext_i32_i64(skip_threads_5087))];\n                        x_5071 = ((volatile __local\n                                   float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                                    (int64_t) 4) +\n                                                            (sext_i32_i64(local_tid_5033) -\n                                                             sext_i32_i64(skip_threads_5087))];\n                    }\n                    // perform operation\n                    {\n                        bool f_5074 = x_5070 || x_5072;\n                        float defunc_1_op_res_5075;\n                        \n                        if (x_5072) {\n                            defunc_1_op_res_5075 = x_5073;\n                        } else {\n                            float defunc_1_op_res_f_res_5076 = x_5071 + x_5073;\n                            \n                            defunc_1_op_res_5075 = defunc_1_op_res_f_res_5076;\n                        }\n         ",
            "               x_5070 = f_5074;\n                        x_5071 = defunc_1_op_res_5075;\n                    }\n                }\n                if (sle32(wave_sizze_5035, skip_threads_5087)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_5087, local_tid_5033 -\n                          squot32(local_tid_5033, 32) * 32) &&\n                    ltid_in_bounds_5086) {\n                    // write result\n                    {\n                        ((volatile __local\n                          bool *) local_mem_5043)[byte_offsets_5037 +\n                                                  sext_i32_i64(local_tid_5033)] =\n                            x_5070;\n                        x_5072 = x_5070;\n                        ((volatile __local\n                          float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                           (int64_t) 4) +\n                                                   sext_i32_i64(local_tid_5033)] =\n                            x_5071;\n                        x_5073 = x_5071;\n                    }\n                }\n                if (sle32(wave_sizze_5035, skip_threads_5087)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_5087 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_5033 - squot32(local_tid_5033, 32) * 32) == 31 &&\n                ltid_in_bounds_5086) {\n                ((volatile __local bool *) local_mem_5043)[byte_offsets_5037 +\n                                                           sext_i32_i64(squot32(local_tid_5033,\n                                                                                32))] =\n                    x_5070;\n                ((volatile __local\n                  float *) local_mem_5043)[squot64(byte_offsets_5038,\n                      ",
            "                             (int64_t) 4) +\n                                           sext_i32_i64(squot32(local_tid_5033,\n                                                                32))] = x_5071;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n        {\n            int32_t skip_threads_5088;\n            \n            // read input for in-block scan\n            {\n                if (squot32(local_tid_5033, 32) == 0 && ltid_in_bounds_5086) {\n                    x_5081 = ((volatile __local\n                               bool *) local_mem_5043)[byte_offsets_5037 +\n                                                       sext_i32_i64(local_tid_5033)];\n                    x_5082 = ((volatile __local\n                               float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                                (int64_t) 4) +\n                                                        sext_i32_i64(local_tid_5033)];\n                    if ((local_tid_5033 - squot32(local_tid_5033, 32) * 32) ==\n                        0) {\n                        x_5079 = x_5081;\n                        x_5080 = x_5082;\n                    }\n                }\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_5088 = 1;\n                while (slt32(skip_threads_5088, 32)) {\n                    if (sle32(skip_threads_5088, local_tid_5033 -\n                              squot32(local_tid_5033, 32) * 32) &&\n                        (squot32(local_tid_5033, 32) == 0 &&\n                         ltid_in_bounds_5086)) {\n                        // read operands\n                        {\n                            x_5079 = ((volatile __local\n                                       bool *) local_mem_5043)[byte_offsets_5037 +\n                                                               (sext_i32_i64(l",
            "ocal_tid_5033) -\n                                                                sext_i32_i64(skip_threads_5088))];\n                            x_5080 = ((volatile __local\n                                       float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                                        (int64_t) 4) +\n                                                                (sext_i32_i64(local_tid_5033) -\n                                                                 sext_i32_i64(skip_threads_5088))];\n                        }\n                        // perform operation\n                        {\n                            bool f_5083 = x_5079 || x_5081;\n                            float defunc_1_op_res_5084;\n                            \n                            if (x_5081) {\n                                defunc_1_op_res_5084 = x_5082;\n                            } else {\n                                float defunc_1_op_res_f_res_5085 = x_5080 +\n                                      x_5082;\n                                \n                                defunc_1_op_res_5084 =\n                                    defunc_1_op_res_f_res_5085;\n                            }\n                            x_5079 = f_5083;\n                            x_5080 = defunc_1_op_res_5084;\n                        }\n                    }\n                    if (sle32(wave_sizze_5035, skip_threads_5088)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_5088, local_tid_5033 -\n                              squot32(local_tid_5033, 32) * 32) &&\n                        (squot32(local_tid_5033, 32) == 0 &&\n                         ltid_in_bounds_5086)) {\n                        // write result\n                        {\n                            ((volatile __local\n                              bool *) local_mem_5043)[byte_offsets_5037 +\n                                       ",
            "               sext_i32_i64(local_tid_5033)] =\n                                x_5079;\n                            x_5081 = x_5079;\n                            ((volatile __local\n                              float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                               (int64_t) 4) +\n                                                       sext_i32_i64(local_tid_5033)] =\n                                x_5080;\n                            x_5082 = x_5080;\n                        }\n                    }\n                    if (sle32(wave_sizze_5035, skip_threads_5088)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_5088 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_5033, 32) == 0 || !ltid_in_bounds_5086)) {\n                // read operands\n                {\n                    x_5072 = x_5070;\n                    x_5073 = x_5071;\n                    x_5070 = ((__local\n                               bool *) local_mem_5043)[byte_offsets_5037 +\n                                                       (sext_i32_i64(squot32(local_tid_5033,\n                                                                             32)) -\n                                                        (int64_t) 1)];\n                    x_5071 = ((__local\n                               float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                                (int64_t) 4) +\n                                                        (sext_i32_i64(squot32(local_tid_5033,\n                                                                              32)) -\n                                                         (int64_t) 1)];\n                }\n                // perform operation\n                {\n                  ",
            "  bool f_5074 = x_5070 || x_5072;\n                    float defunc_1_op_res_5075;\n                    \n                    if (x_5072) {\n                        defunc_1_op_res_5075 = x_5073;\n                    } else {\n                        float defunc_1_op_res_f_res_5076 = x_5071 + x_5073;\n                        \n                        defunc_1_op_res_5075 = defunc_1_op_res_f_res_5076;\n                    }\n                    x_5070 = f_5074;\n                    x_5071 = defunc_1_op_res_5075;\n                }\n                // write final result\n                {\n                    ((__local bool *) local_mem_5043)[byte_offsets_5037 +\n                                                      sext_i32_i64(local_tid_5033)] =\n                        x_5070;\n                    ((__local\n                      float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                       (int64_t) 4) +\n                                               sext_i32_i64(local_tid_5033)] =\n                        x_5071;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_5033, 32) == 0) {\n                ((__local bool *) local_mem_5043)[byte_offsets_5037 +\n                                                  sext_i32_i64(local_tid_5033)] =\n                    x_5072;\n                ((__local float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                           (int64_t) 4) +\n                                                   sext_i32_i64(local_tid_5033)] =\n                    x_5073;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_5033 == 0) {\n            acc_5077 = ((__local bool *) local_mem_5043)[byte_offsets_5037 +\n                                                         (segscan_group_sizze_4829 -\n          ",
            "                                                (int64_t) 1)];\n            acc_5078 = ((__local\n                         float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                          (int64_t) 4) +\n                                                  (segscan_group_sizze_4829 -\n                                                   (int64_t) 1)];\n        } else {\n            acc_5077 = ((__local bool *) local_mem_5043)[byte_offsets_5037 +\n                                                         (sext_i32_i64(local_tid_5033) -\n                                                          (int64_t) 1)];\n            acc_5078 = ((__local\n                         float *) local_mem_5043)[squot64(byte_offsets_5038,\n                                                          (int64_t) 4) +\n                                                  (sext_i32_i64(local_tid_5033) -\n                                                   (int64_t) 1)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n    bool prefix_5089;\n    \n    prefix_5089 = 0;\n    \n    float prefix_5090;\n    \n    prefix_5090 = 0.0F;\n    // Perform lookback\n    {\n        if (dynamic_id_5054 == (int64_t) 0 && local_tid_5033 == 0) {\n            ((volatile __global bool *) incprefixes_mem_5026)[dynamic_id_5054] =\n                acc_5077;\n            ((volatile __global\n              float *) incprefixes_mem_5030)[dynamic_id_5054] = acc_5078;\n            mem_fence_global();\n            ((volatile __global\n              int8_t *) status_flags_mem_5022)[dynamic_id_5054] = (int8_t) 2;\n            acc_5077 = 0;\n            acc_5078 = 0.0F;\n        }\n        if (!(dynamic_id_5054 == (int64_t) 0) && slt32(local_tid_5033,\n                                                       wave_sizze_5035)) {\n            if (local_tid_5033 == 0) {\n                ((volatile __global\n                  bool *) aggregates_mem_5024)[dynamic_id_5054] = acc_5077;\n                ((volatile __global\n       ",
            "           float *) aggregates_mem_5028)[dynamic_id_5054] = acc_5078;\n                mem_fence_global();\n                ((volatile __global\n                  int8_t *) status_flags_mem_5022)[dynamic_id_5054] =\n                    (int8_t) 1;\n                ((__local int8_t *) local_mem_5043)[(int64_t) 0] = ((__global\n                                                                     int8_t *) status_flags_mem_5022)[dynamic_id_5054 -\n                                                                                                      (int64_t) 1];\n            }\n            mem_fence_local();\n            \n            int8_t status_5091;\n            \n            status_5091 = ((__local int8_t *) local_mem_5043)[(int64_t) 0];\n            if (status_5091 == (int8_t) 2) {\n                if (local_tid_5033 == 0) {\n                    prefix_5089 = ((volatile __global\n                                    bool *) incprefixes_mem_5026)[dynamic_id_5054 -\n                                                                  (int64_t) 1];\n                    prefix_5090 = ((volatile __global\n                                    float *) incprefixes_mem_5030)[dynamic_id_5054 -\n                                                                   (int64_t) 1];\n                }\n            } else {\n                int32_t readOffset_5092 = dynamic_id_5054 -\n                        sext_i32_i64(wave_sizze_5035);\n                \n                while (slt32(wave_sizze_5035 * -1, readOffset_5092)) {\n                    int32_t read_i_5093 = readOffset_5092 + local_tid_5033;\n                    bool aggr_5094 = 0;\n                    float aggr_5095 = 0.0F;\n                    int8_t flag_5096 = (int8_t) 0;\n                    int8_t used_5097 = (int8_t) 0;\n                    \n                    if (sle32(0, read_i_5093)) {\n                        flag_5096 = ((volatile __global\n                                      int8_t *) status_flags_mem_5022)[sext_i32_i64(read_i_5093)];\n        ",
            "                if (flag_5096 == (int8_t) 2) {\n                            aggr_5094 = ((volatile __global\n                                          bool *) incprefixes_mem_5026)[sext_i32_i64(read_i_5093)];\n                            aggr_5095 = ((volatile __global\n                                          float *) incprefixes_mem_5030)[sext_i32_i64(read_i_5093)];\n                        } else {\n                            if (flag_5096 == (int8_t) 1) {\n                                aggr_5094 = ((volatile __global\n                                              bool *) aggregates_mem_5024)[sext_i32_i64(read_i_5093)];\n                                aggr_5095 = ((volatile __global\n                                              float *) aggregates_mem_5028)[sext_i32_i64(read_i_5093)];\n                                used_5097 = (int8_t) 1;\n                            }\n                        }\n                    }\n                    ((__local bool *) local_mem_5043)[warp_byte_offset_5040 +\n                                                      sext_i32_i64(local_tid_5033)] =\n                        aggr_5094;\n                    ((__local\n                      float *) local_mem_5043)[squot64(warp_byte_offset_5041,\n                                                       (int64_t) 4) +\n                                               sext_i32_i64(local_tid_5033)] =\n                        aggr_5095;\n                    \n                    int8_t tmp_5098;\n                    \n                    tmp_5098 = flag_5096 | used_5097 << (int8_t) 2;\n                    ((__local\n                      int8_t *) local_mem_5043)[sext_i32_i64(local_tid_5033)] =\n                        tmp_5098;\n                    mem_fence_local();\n                    flag_5096 = ((volatile __local\n                                  int8_t *) local_mem_5043)[sext_i32_i64(wave_sizze_5035) -\n                                                            (int64_t) 1];\n                    if (local_tid",
            "_5033 == 0) {\n                        if (!(flag_5096 == (int8_t) 2)) {\n                            bool x_5099 = 0;\n                            float x_5100 = 0.0F;\n                            bool x_5101;\n                            float x_5102;\n                            int8_t flag1_5106;\n                            \n                            flag1_5106 = (int8_t) 0;\n                            \n                            int8_t flag2_5107;\n                            int8_t used1_5108;\n                            \n                            used1_5108 = (int8_t) 0;\n                            \n                            int8_t used2_5109;\n                            \n                            for (int32_t i_5110 = 0; i_5110 < wave_sizze_5035;\n                                 i_5110++) {\n                                flag2_5107 = ((__local\n                                               int8_t *) local_mem_5043)[sext_i32_i64(i_5110)];\n                                used2_5109 = lshr8(flag2_5107, (int8_t) 2);\n                                flag2_5107 = flag2_5107 & (int8_t) 3;\n                                x_5101 = ((__local\n                                           bool *) local_mem_5043)[warp_byte_offset_5040 +\n                                                                   sext_i32_i64(i_5110)];\n                                x_5102 = ((__local\n                                           float *) local_mem_5043)[squot64(warp_byte_offset_5041,\n                                                                            (int64_t) 4) +\n                                                                    sext_i32_i64(i_5110)];\n                                if (!(flag2_5107 == (int8_t) 1)) {\n                                    flag1_5106 = flag2_5107;\n                                    used1_5108 = used2_5109;\n                                    x_5099 = x_5101;\n                                    x_5100 = x_5102;\n                                ",
            "} else {\n                                    used1_5108 += used2_5109;\n                                    \n                                    bool f_5103;\n                                    \n                                    f_5103 = x_5099 || x_5101;\n                                    \n                                    float defunc_1_op_res_5104;\n                                    \n                                    if (x_5101) {\n                                        defunc_1_op_res_5104 = x_5102;\n                                    } else {\n                                        float defunc_1_op_res_f_res_5105 =\n                                              x_5100 + x_5102;\n                                        \n                                        defunc_1_op_res_5104 =\n                                            defunc_1_op_res_f_res_5105;\n                                    }\n                                    x_5099 = f_5103;\n                                    x_5100 = defunc_1_op_res_5104;\n                                }\n                            }\n                            flag_5096 = flag1_5106;\n                            used_5097 = used1_5108;\n                            aggr_5094 = x_5099;\n                            aggr_5095 = x_5100;\n                        } else {\n                            aggr_5094 = ((__local\n                                          bool *) local_mem_5043)[warp_byte_offset_5040 +\n                                                                  (sext_i32_i64(wave_sizze_5035) -\n                                                                   (int64_t) 1)];\n                            aggr_5095 = ((__local\n                                          float *) local_mem_5043)[squot64(warp_byte_offset_5041,\n                                                                           (int64_t) 4) +\n                                                                   (sext_i32_i64(wave_sizze_5035) -\n           ",
            "                                                         (int64_t) 1)];\n                        }\n                        if (flag_5096 == (int8_t) 2) {\n                            readOffset_5092 = wave_sizze_5035 * -1;\n                        } else {\n                            readOffset_5092 -= zext_i8_i32(used_5097);\n                        }\n                        ((__local int32_t *) local_mem_5043)[(int64_t) 0] =\n                            readOffset_5092;\n                        \n                        bool x_5111;\n                        \n                        x_5111 = aggr_5094;\n                        \n                        float x_5112;\n                        \n                        x_5112 = aggr_5095;\n                        \n                        bool x_5113;\n                        \n                        x_5113 = prefix_5089;\n                        \n                        float x_5114;\n                        \n                        x_5114 = prefix_5090;\n                        \n                        bool f_5115;\n                        \n                        f_5115 = x_5111 || x_5113;\n                        \n                        float defunc_1_op_res_5116;\n                        \n                        if (x_5113) {\n                            defunc_1_op_res_5116 = x_5114;\n                        } else {\n                            float defunc_1_op_res_f_res_5117 = x_5112 + x_5114;\n                            \n                            defunc_1_op_res_5116 = defunc_1_op_res_f_res_5117;\n                        }\n                        prefix_5089 = f_5115;\n                        prefix_5090 = defunc_1_op_res_5116;\n                    }\n                    mem_fence_local();\n                    readOffset_5092 = ((__local\n                                        int32_t *) local_mem_5043)[(int64_t) 0];\n                }\n            }\n            if (local_tid_5033 == 0) {\n                bool x_5118 = prefix_5089;\n    ",
            "            float x_5119 = prefix_5090;\n                bool x_5120 = acc_5077;\n                float x_5121 = acc_5078;\n                bool f_5122 = x_5118 || x_5120;\n                float defunc_1_op_res_5123;\n                \n                if (x_5120) {\n                    defunc_1_op_res_5123 = x_5121;\n                } else {\n                    float defunc_1_op_res_f_res_5124 = x_5119 + x_5121;\n                    \n                    defunc_1_op_res_5123 = defunc_1_op_res_f_res_5124;\n                }\n                ((volatile __global\n                  bool *) incprefixes_mem_5026)[dynamic_id_5054] = f_5122;\n                ((volatile __global\n                  float *) incprefixes_mem_5030)[dynamic_id_5054] =\n                    defunc_1_op_res_5123;\n                mem_fence_global();\n                ((volatile __global\n                  int8_t *) status_flags_mem_5022)[dynamic_id_5054] =\n                    (int8_t) 2;\n                ((__local bool *) local_mem_5043)[warp_byte_offset_5040] =\n                    prefix_5089;\n                ((__local\n                  float *) local_mem_5043)[squot64(warp_byte_offset_5041,\n                                                   (int64_t) 4)] = prefix_5090;\n                acc_5077 = 0;\n                acc_5078 = 0.0F;\n            }\n        }\n        if (!(dynamic_id_5054 == (int64_t) 0)) {\n            barrier(CLK_LOCAL_MEM_FENCE);\n            prefix_5089 = ((__local\n                            bool *) local_mem_5043)[warp_byte_offset_5040];\n            prefix_5090 = ((__local\n                            float *) local_mem_5043)[squot64(warp_byte_offset_5041,\n                                                             (int64_t) 4)];\n            barrier(CLK_LOCAL_MEM_FENCE);\n        }\n    }\n    // Distribute results\n    {\n        bool x_5125;\n        bool x_5127;\n        bool x_5132;\n        \n        x_5132 = prefix_5089;\n        \n        bool x_5134;\n        \n        x_5134 = acc_5077;\n        \n        fl",
            "oat x_5126;\n        float x_5128;\n        float x_5133;\n        \n        x_5133 = prefix_5090;\n        \n        float x_5135;\n        \n        x_5135 = acc_5078;\n        \n        bool f_5136;\n        \n        f_5136 = x_5132 || x_5134;\n        \n        float defunc_1_op_res_5137;\n        \n        if (x_5134) {\n            defunc_1_op_res_5137 = x_5135;\n        } else {\n            float defunc_1_op_res_f_res_5138 = x_5133 + x_5135;\n            \n            defunc_1_op_res_5137 = defunc_1_op_res_f_res_5138;\n        }\n        x_5125 = f_5136;\n        x_5126 = defunc_1_op_res_5137;\n        for (int64_t i_5139 = 0; i_5139 < (int64_t) 9; i_5139++) {\n            x_5127 = private_mem_5056[i_5139];\n            x_5128 = private_mem_5058[i_5139];\n            \n            bool f_5129;\n            \n            f_5129 = x_5125 || x_5127;\n            \n            float defunc_1_op_res_5130;\n            \n            if (x_5127) {\n                defunc_1_op_res_5130 = x_5128;\n            } else {\n                float defunc_1_op_res_f_res_5131 = x_5126 + x_5128;\n                \n                defunc_1_op_res_5130 = defunc_1_op_res_f_res_5131;\n            }\n            private_mem_5056[i_5139] = f_5129;\n            private_mem_5058[i_5139] = defunc_1_op_res_5130;\n        }\n    }\n    // Transpose scan output\n    {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5140 = 0; i_5140 < (int64_t) 9; i_5140++) {\n            int64_t sharedIdx_5141 = sext_i32_i64(local_tid_5033 * 9) + i_5140;\n            \n            ((__local bool *) local_mem_5043)[sharedIdx_5141] =\n                private_mem_5056[i_5140];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5142 = 0; i_5142 < (int64_t) 9; i_5142++) {\n            int32_t sharedIdx_5143 = local_tid_5033 +\n                    sext_i64_i32(segscan_group_sizze_4829 * i_5142);\n            \n            private_mem_5056[i_5142] = ((__local\n                                         bool *) local_mem_5043)[sext_i32_",
            "i64(sharedIdx_5143)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5144 = 0; i_5144 < (int64_t) 9; i_5144++) {\n            int64_t sharedIdx_5145 = sext_i32_i64(local_tid_5033 * 9) + i_5144;\n            \n            ((__local float *) local_mem_5043)[sharedIdx_5145] =\n                private_mem_5058[i_5144];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int64_t i_5146 = 0; i_5146 < (int64_t) 9; i_5146++) {\n            int32_t sharedIdx_5147 = local_tid_5033 +\n                    sext_i64_i32(segscan_group_sizze_4829 * i_5146);\n            \n            private_mem_5058[i_5146] = ((__local\n                                         float *) local_mem_5043)[sext_i32_i64(sharedIdx_5147)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    // Write block scan results to global memory\n    {\n        for (int64_t i_5148 = 0; i_5148 < (int64_t) 9; i_5148++) {\n            int64_t gtid_4833 = blockOff_5055 + segscan_group_sizze_4829 *\n                    i_5148 + sext_i32_i64(local_tid_5033);\n            \n            if (slt64(gtid_4833, n_4772)) {\n                ((__global bool *) mem_4901)[gtid_4833] =\n                    private_mem_5056[i_5148];\n            }\n        }\n        for (int64_t i_5149 = 0; i_5149 < (int64_t) 9; i_5149++) {\n            int64_t gtid_4833 = blockOff_5055 + segscan_group_sizze_4829 *\n                    i_5149 + sext_i32_i64(local_tid_5033);\n            \n            if (slt64(gtid_4833, n_4772)) {\n                ((__global float *) mem_4903)[gtid_4833] =\n                    private_mem_5058[i_5149];\n            }\n        }\n    }\n    // If this is the last block, reset the dynamicId\n    {\n        if (dynamic_id_5054 == sdiv_up64(n_4772, segscan_group_sizze_4829 *\n                                         (int64_t) 9) - (int64_t) 1) {\n            ((__global int32_t *) mainziid_counter_mem_5020)[(int64_t) 0] = 0;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segscan_group_sizze_4829\n}\n",
            NULL};
static const char *size_names[] = {"builtin#replicate_bool.group_size_5010",
                                   "builtin#replicate_i8.group_size_4930",
                                   "main.segmap_group_size_4824",
                                   "main.segmap_group_size_4864",
                                   "main.segscan_group_size_4814",
                                   "main.segscan_group_size_4828",
                                   "main.segscan_num_groups_4816",
                                   "main.segscan_num_groups_4830"};
static const char *size_vars[] = {"builtinzhreplicate_boolzigroup_sizze_5010",
                                  "builtinzhreplicate_i8zigroup_sizze_4930",
                                  "mainzisegmap_group_sizze_4824",
                                  "mainzisegmap_group_sizze_4864",
                                  "mainzisegscan_group_sizze_4814",
                                  "mainzisegscan_group_sizze_4828",
                                  "mainzisegscan_num_groups_4816",
                                  "mainzisegscan_num_groups_4830"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "num_groups", "num_groups"};
int futhark_get_num_sizes(void)
{
    return 8;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    int64_t builtinzhreplicate_boolzigroup_sizze_5010;
    int64_t builtinzhreplicate_i8zigroup_sizze_4930;
    int64_t mainzisegmap_group_sizze_4824;
    int64_t mainzisegmap_group_sizze_4864;
    int64_t mainzisegscan_group_sizze_4814;
    int64_t mainzisegscan_group_sizze_4828;
    int64_t mainzisegscan_num_groups_4816;
    int64_t mainzisegscan_num_groups_4830;
} ;
struct futhark_context_config {
    struct cuda_config cu_cfg;
    int profiling;
    int64_t sizes[8];
    int num_nvrtc_opts;
    const char **nvrtc_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->profiling = 0;
    cfg->num_nvrtc_opts = 0;
    cfg->nvrtc_opts = (const char **) malloc(sizeof(const char *));
    cfg->nvrtc_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cuda_config_init(&cfg->cu_cfg, 8, size_names, size_vars, cfg->sizes,
                     size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->nvrtc_opts);
    free(cfg);
}
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->nvrtc_opts[cfg->num_nvrtc_opts] = opt;
    cfg->num_nvrtc_opts++;
    cfg->nvrtc_opts = (const char **) realloc(cfg->nvrtc_opts,
                                              (cfg->num_nvrtc_opts + 1) *
                                              sizeof(const char *));
    cfg->nvrtc_opts[cfg->num_nvrtc_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->cu_cfg.logging = cfg->cu_cfg.debugging = flag;
}
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->profiling = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->cu_cfg.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->cu_cfg, s);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->cu_cfg.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->cu_cfg.load_program_from = path;
}
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path)
{
    cfg->cu_cfg.dump_ptx_to = path;
}
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path)
{
    cfg->cu_cfg.load_ptx_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->cu_cfg.default_block_size = size;
    cfg->cu_cfg.default_block_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->cu_cfg.default_grid_size = num;
    cfg->cu_cfg.default_grid_size_changed = 1;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->cu_cfg.default_tile_size = size;
    cfg->cu_cfg.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->cu_cfg.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 8; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    if (strcmp(size_name, "default_group_size") == 0) {
        cfg->cu_cfg.default_block_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_num_groups") == 0) {
        cfg->cu_cfg.default_grid_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_threshold") == 0) {
        cfg->cu_cfg.default_threshold = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_tile_size") == 0) {
        cfg->cu_cfg.default_tile_size = size_value;
        return 0;
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    int profiling_paused;
    int logging;
    lock_t lock;
    char *error;
    FILE *log;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct {
        int dummy;
    } constants;
    struct memblock_device mainziid_counter_mem_4915;
    struct memblock_device mainziid_counter_mem_5020;
    CUfunction builtinzhreplicate_boolzireplicate_5007;
    int64_t builtinzhreplicate_boolzireplicate_5007_total_runtime;
    int builtinzhreplicate_boolzireplicate_5007_runs;
    CUfunction builtinzhreplicate_i8zireplicate_4927;
    int64_t builtinzhreplicate_i8zireplicate_4927_total_runtime;
    int builtinzhreplicate_i8zireplicate_4927_runs;
    CUfunction mainzisegmap_4822;
    int64_t mainzisegmap_4822_total_runtime;
    int mainzisegmap_4822_runs;
    CUfunction mainzisegmap_4862;
    int64_t mainzisegmap_4862_total_runtime;
    int mainzisegmap_4862_runs;
    CUfunction mainzisegscan_4820;
    int64_t mainzisegscan_4820_total_runtime;
    int mainzisegscan_4820_runs;
    CUfunction mainzisegscan_4834;
    int64_t mainzisegscan_4834_total_runtime;
    int mainzisegscan_4834_runs;
    int64_t copy_dev_to_dev_total_runtime;
    int copy_dev_to_dev_runs;
    int64_t copy_dev_to_host_total_runtime;
    int copy_dev_to_host_runs;
    int64_t copy_host_to_dev_total_runtime;
    int copy_host_to_dev_runs;
    int64_t copy_scalar_to_dev_total_runtime;
    int copy_scalar_to_dev_runs;
    int64_t copy_scalar_from_dev_total_runtime;
    int copy_scalar_from_dev_runs;
    CUdeviceptr global_failure;
    CUdeviceptr global_failure_args;
    struct cuda_context cuda;
    struct sizes sizes;
    int32_t failure_is_an_option;
    int total_runs;
    long total_runtime;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->debugging = ctx->detail_memory = cfg->cu_cfg.debugging;
    ctx->profiling = cfg->profiling;
    ctx->profiling_paused = 0;
    ctx->logging = cfg->cu_cfg.logging;
    ctx->error = NULL;
    ctx->log = stderr;
    ctx->cuda.profiling_records_capacity = 200;
    ctx->cuda.profiling_records_used = 0;
    ctx->cuda.profiling_records = malloc(ctx->cuda.profiling_records_capacity *
        sizeof(struct profiling_record));
    ctx->cuda.cfg = cfg->cu_cfg;
    create_lock(&ctx->lock);
    ctx->failure_is_an_option = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    cuda_setup(&ctx->cuda, cuda_program, cfg->nvrtc_opts);
    
    int32_t no_error = -1;
    
    CUDA_SUCCEED(cuMemAlloc(&ctx->global_failure, sizeof(no_error)));
    CUDA_SUCCEED(cuMemcpyHtoD(ctx->global_failure, &no_error,
                              sizeof(no_error)));
    // The +1 is to avoid zero-byte allocations.
    CUDA_SUCCEED(cuMemAlloc(&ctx->global_failure_args, sizeof(int64_t) * (2 +
                                                                          1)));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->builtinzhreplicate_boolzireplicate_5007,
                                     ctx->cuda.module,
                                     "builtinzhreplicate_boolzireplicate_5007"));
    ctx->builtinzhreplicate_boolzireplicate_5007_total_runtime = 0;
    ctx->builtinzhreplicate_boolzireplicate_5007_runs = 0;
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->builtinzhreplicate_i8zireplicate_4927,
                                     ctx->cuda.module,
                                     "builtinzhreplicate_i8zireplicate_4927"));
    ctx->builtinzhreplicate_i8zireplicate_4927_total_runtime = 0;
    ctx->builtinzhreplicate_i8zireplicate_4927_runs = 0;
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->mainzisegmap_4822, ctx->cuda.module,
                                     "mainzisegmap_4822"));
    ctx->mainzisegmap_4822_total_runtime = 0;
    ctx->mainzisegmap_4822_runs = 0;
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->mainzisegmap_4862, ctx->cuda.module,
                                     "mainzisegmap_4862"));
    ctx->mainzisegmap_4862_total_runtime = 0;
    ctx->mainzisegmap_4862_runs = 0;
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->mainzisegscan_4820, ctx->cuda.module,
                                     "mainzisegscan_4820"));
    ctx->mainzisegscan_4820_total_runtime = 0;
    ctx->mainzisegscan_4820_runs = 0;
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->mainzisegscan_4834, ctx->cuda.module,
                                     "mainzisegscan_4834"));
    ctx->mainzisegscan_4834_total_runtime = 0;
    ctx->mainzisegscan_4834_runs = 0;
    ctx->copy_dev_to_dev_total_runtime = 0;
    ctx->copy_dev_to_dev_runs = 0;
    ctx->copy_dev_to_host_total_runtime = 0;
    ctx->copy_dev_to_host_runs = 0;
    ctx->copy_host_to_dev_total_runtime = 0;
    ctx->copy_host_to_dev_runs = 0;
    ctx->copy_scalar_to_dev_total_runtime = 0;
    ctx->copy_scalar_to_dev_runs = 0;
    ctx->copy_scalar_from_dev_total_runtime = 0;
    ctx->copy_scalar_from_dev_runs = 0;
    {
        ctx->mainziid_counter_mem_4915.references = NULL;
        ctx->mainziid_counter_mem_4915.size = 0;
        CUDA_SUCCEED(cuMemAlloc(&ctx->mainziid_counter_mem_4915.mem, (1 >
                                                                      0 ? 1 : 1) *
                                sizeof(int32_t)));
        if (1 > 0)
            CUDA_SUCCEED(cuMemcpyHtoD(ctx->mainziid_counter_mem_4915.mem,
                                      mainziid_counter_mem_realtype_5164, 1 *
                                      sizeof(int32_t)));
    }
    {
        ctx->mainziid_counter_mem_5020.references = NULL;
        ctx->mainziid_counter_mem_5020.size = 0;
        CUDA_SUCCEED(cuMemAlloc(&ctx->mainziid_counter_mem_5020.mem, (1 >
                                                                      0 ? 1 : 1) *
                                sizeof(int32_t)));
        if (1 > 0)
            CUDA_SUCCEED(cuMemcpyHtoD(ctx->mainziid_counter_mem_5020.mem,
                                      mainziid_counter_mem_realtype_5181, 1 *
                                      sizeof(int32_t)));
    }
    ctx->sizes.builtinzhreplicate_boolzigroup_sizze_5010 = cfg->sizes[0];
    ctx->sizes.builtinzhreplicate_i8zigroup_sizze_4930 = cfg->sizes[1];
    ctx->sizes.mainzisegmap_group_sizze_4824 = cfg->sizes[2];
    ctx->sizes.mainzisegmap_group_sizze_4864 = cfg->sizes[3];
    ctx->sizes.mainzisegscan_group_sizze_4814 = cfg->sizes[4];
    ctx->sizes.mainzisegscan_group_sizze_4828 = cfg->sizes[5];
    ctx->sizes.mainzisegscan_num_groups_4816 = cfg->sizes[6];
    ctx->sizes.mainzisegscan_num_groups_4830 = cfg->sizes[7];
    init_constants(ctx);
    // Clear the free list of any deallocations that occurred while initialising constants.
    CUDA_SUCCEED(cuda_free_all(&ctx->cuda));
    futhark_context_sync(ctx);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_constants(ctx);
    cuda_cleanup(&ctx->cuda);
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    CUDA_SUCCEED(cuCtxSynchronize());
    if (ctx->failure_is_an_option) {
        int32_t failure_idx;
        
        CUDA_SUCCEED(cuMemcpyDtoH(&failure_idx, ctx->global_failure,
                                  sizeof(int32_t)));
        ctx->failure_is_an_option = 0;
        if (failure_idx >= 0) {
            int32_t no_failure = -1;
            
            CUDA_SUCCEED(cuMemcpyHtoD(ctx->global_failure, &no_failure,
                                      sizeof(int32_t)));
            
            int64_t args[2 + 1];
            
            CUDA_SUCCEED(cuMemcpyDtoH(&args, ctx->global_failure_args,
                                      sizeof(args)));
            switch (failure_idx) {
                
              case 0:
                {
                    ctx->error =
                        msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  spMVmult-flat.fut:107:32-37\n   #1  spMVmult-flat.fut:107:15-46\n   #2  spMVmult-flat.fut:121:3-46\n   #3  spMVmult-flat.fut:118:1-121:46\n",
                                  args[0], args[1]);
                    break;
                }
                
              case 1:
                {
                    ctx->error =
                        msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  spMVmult-flat.fut:113:30-36\n   #1  spMVmult-flat.fut:113:13-47\n   #2  spMVmult-flat.fut:121:3-46\n   #3  spMVmult-flat.fut:118:1-121:46\n",
                                  args[0], args[1]);
                    break;
                }
            }
            return 1;
        }
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    return 0;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            CUDA_SUCCEED(cuda_free(&ctx->cuda, block->mem, desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "space 'device'",
                      ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(ctx->log,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(ctx->log, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(ctx->log, ".\n");
    CUDA_SUCCEED(cuda_alloc(&ctx->cuda, size, desc, &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "default space",
                      ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(ctx->log,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(ctx->log, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(ctx->log, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
char *futhark_context_report(struct futhark_context *ctx)
{
    struct str_builder builder;
    
    str_builder_init(&builder);
    if (ctx->detail_memory || ctx->profiling || ctx->logging) {
        str_builder(&builder,
                    "Peak memory usage for space 'device': %lld bytes.\n",
                    (long long) ctx->peak_mem_usage_device);
        { }
    }
    if (ctx->profiling) {
        CUDA_SUCCEED(cuda_tally_profiling_records(&ctx->cuda));
        str_builder(&builder,
                    "copy_dev_to_dev                       ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_dev_to_dev_runs,
                    (long) ctx->copy_dev_to_dev_total_runtime /
                    (ctx->copy_dev_to_dev_runs !=
                     0 ? ctx->copy_dev_to_dev_runs : 1),
                    (long) ctx->copy_dev_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_dev_runs;
        str_builder(&builder,
                    "copy_dev_to_host                      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_dev_to_host_runs,
                    (long) ctx->copy_dev_to_host_total_runtime /
                    (ctx->copy_dev_to_host_runs !=
                     0 ? ctx->copy_dev_to_host_runs : 1),
                    (long) ctx->copy_dev_to_host_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_host_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_host_runs;
        str_builder(&builder,
                    "copy_host_to_dev                      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_host_to_dev_runs,
                    (long) ctx->copy_host_to_dev_total_runtime /
                    (ctx->copy_host_to_dev_runs !=
                     0 ? ctx->copy_host_to_dev_runs : 1),
                    (long) ctx->copy_host_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_host_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_host_to_dev_runs;
        str_builder(&builder,
                    "copy_scalar_to_dev                    ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_scalar_to_dev_runs,
                    (long) ctx->copy_scalar_to_dev_total_runtime /
                    (ctx->copy_scalar_to_dev_runs !=
                     0 ? ctx->copy_scalar_to_dev_runs : 1),
                    (long) ctx->copy_scalar_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_to_dev_runs;
        str_builder(&builder,
                    "copy_scalar_from_dev                  ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_scalar_from_dev_runs,
                    (long) ctx->copy_scalar_from_dev_total_runtime /
                    (ctx->copy_scalar_from_dev_runs !=
                     0 ? ctx->copy_scalar_from_dev_runs : 1),
                    (long) ctx->copy_scalar_from_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_from_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_from_dev_runs;
        str_builder(&builder,
                    "builtin#replicate_bool.replicate_5007 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->builtinzhreplicate_boolzireplicate_5007_runs,
                    (long) ctx->builtinzhreplicate_boolzireplicate_5007_total_runtime /
                    (ctx->builtinzhreplicate_boolzireplicate_5007_runs !=
                     0 ? ctx->builtinzhreplicate_boolzireplicate_5007_runs : 1),
                    (long) ctx->builtinzhreplicate_boolzireplicate_5007_total_runtime);
        ctx->total_runtime +=
            ctx->builtinzhreplicate_boolzireplicate_5007_total_runtime;
        ctx->total_runs += ctx->builtinzhreplicate_boolzireplicate_5007_runs;
        str_builder(&builder,
                    "builtin#replicate_i8.replicate_4927   ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->builtinzhreplicate_i8zireplicate_4927_runs,
                    (long) ctx->builtinzhreplicate_i8zireplicate_4927_total_runtime /
                    (ctx->builtinzhreplicate_i8zireplicate_4927_runs !=
                     0 ? ctx->builtinzhreplicate_i8zireplicate_4927_runs : 1),
                    (long) ctx->builtinzhreplicate_i8zireplicate_4927_total_runtime);
        ctx->total_runtime +=
            ctx->builtinzhreplicate_i8zireplicate_4927_total_runtime;
        ctx->total_runs += ctx->builtinzhreplicate_i8zireplicate_4927_runs;
        str_builder(&builder,
                    "main.segmap_4822                      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainzisegmap_4822_runs,
                    (long) ctx->mainzisegmap_4822_total_runtime /
                    (ctx->mainzisegmap_4822_runs !=
                     0 ? ctx->mainzisegmap_4822_runs : 1),
                    (long) ctx->mainzisegmap_4822_total_runtime);
        ctx->total_runtime += ctx->mainzisegmap_4822_total_runtime;
        ctx->total_runs += ctx->mainzisegmap_4822_runs;
        str_builder(&builder,
                    "main.segmap_4862                      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainzisegmap_4862_runs,
                    (long) ctx->mainzisegmap_4862_total_runtime /
                    (ctx->mainzisegmap_4862_runs !=
                     0 ? ctx->mainzisegmap_4862_runs : 1),
                    (long) ctx->mainzisegmap_4862_total_runtime);
        ctx->total_runtime += ctx->mainzisegmap_4862_total_runtime;
        ctx->total_runs += ctx->mainzisegmap_4862_runs;
        str_builder(&builder,
                    "main.segscan_4820                     ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainzisegscan_4820_runs,
                    (long) ctx->mainzisegscan_4820_total_runtime /
                    (ctx->mainzisegscan_4820_runs !=
                     0 ? ctx->mainzisegscan_4820_runs : 1),
                    (long) ctx->mainzisegscan_4820_total_runtime);
        ctx->total_runtime += ctx->mainzisegscan_4820_total_runtime;
        ctx->total_runs += ctx->mainzisegscan_4820_runs;
        str_builder(&builder,
                    "main.segscan_4834                     ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->mainzisegscan_4834_runs,
                    (long) ctx->mainzisegscan_4834_total_runtime /
                    (ctx->mainzisegscan_4834_runs !=
                     0 ? ctx->mainzisegscan_4834_runs : 1),
                    (long) ctx->mainzisegscan_4834_total_runtime);
        ctx->total_runtime += ctx->mainzisegscan_4834_total_runtime;
        ctx->total_runs += ctx->mainzisegscan_4834_runs;
        str_builder(&builder, "%d operations with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
    return builder.str;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f)
{
    ctx->log = f;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    ctx->peak_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    if (ctx->error == NULL)
        CUDA_SUCCEED(cuda_free_all(&ctx->cuda));
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return ctx->error != NULL;
}
static int futrts_builtinzhreplicate_bool(struct futhark_context *ctx,
                                          struct memblock_device mem_5003,
                                          int32_t num_elems_5004,
                                          bool val_5005);
static int futrts_builtinzhreplicate_i8(struct futhark_context *ctx,
                                        struct memblock_device mem_4923,
                                        int32_t num_elems_4924,
                                        int8_t val_4925);
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_5163,
                       struct memblock_device mat_inds_mem_4889,
                       struct memblock_device mat_vals_mem_4890,
                       struct memblock_device shp_mem_4891,
                       struct memblock_device vct_mem_4892, int64_t n_4772,
                       int64_t m_4773, int64_t implz2080U_4774);
static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static int futrts_builtinzhreplicate_bool(struct futhark_context *ctx,
                                          struct memblock_device mem_5003,
                                          int32_t num_elems_5004, bool val_5005)
{
    (void) ctx;
    
    int err = 0;
    int64_t group_sizze_5010;
    
    group_sizze_5010 = ctx->sizes.builtinzhreplicate_boolzigroup_sizze_5010;
    
    int64_t num_groups_5011;
    
    num_groups_5011 = sdiv_up64(num_elems_5004, group_sizze_5010);
    
    CUdeviceptr kernel_arg_5158 = mem_5003.mem;
    
    if ((((((1 && num_groups_5011 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_5010 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 >= 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 >= 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_5011;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_5155[] = {&num_elems_5004, &val_5005,
                                    &kernel_arg_5158};
        int64_t time_start_5156 = 0, time_end_5157 = 0;
        
        if (ctx->debugging) {
            fprintf(ctx->log, "Launching %s with grid size (",
                    "builtin#replicate_bool.replicate_5007");
            fprintf(ctx->log, "%ld", (long) num_groups_5011);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ") and block size (");
            fprintf(ctx->log, "%ld", (long) group_sizze_5010);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ").\n");
            time_start_5156 = get_wall_time();
        }
        
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda,
                                      &ctx->builtinzhreplicate_boolzireplicate_5007_runs,
                                      &ctx->builtinzhreplicate_boolzireplicate_5007_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->builtinzhreplicate_boolzireplicate_5007,
                                    grid[0], grid[1], grid[2], group_sizze_5010,
                                    1, 1, 0, NULL, kernel_args_5155, NULL));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_5157 = get_wall_time();
            fprintf(ctx->log, "Kernel %s runtime: %ldus\n",
                    "builtin#replicate_bool.replicate_5007", time_end_5157 -
                    time_start_5156);
        }
    }
    
  cleanup:
    { }
    return err;
}
static int futrts_builtinzhreplicate_i8(struct futhark_context *ctx,
                                        struct memblock_device mem_4923,
                                        int32_t num_elems_4924, int8_t val_4925)
{
    (void) ctx;
    
    int err = 0;
    int64_t group_sizze_4930;
    
    group_sizze_4930 = ctx->sizes.builtinzhreplicate_i8zigroup_sizze_4930;
    
    int64_t num_groups_4931;
    
    num_groups_4931 = sdiv_up64(num_elems_4924, group_sizze_4930);
    
    CUdeviceptr kernel_arg_5162 = mem_4923.mem;
    
    if ((((((1 && num_groups_4931 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_4930 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 >= 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 >= 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_4931;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_5159[] = {&num_elems_4924, &val_4925,
                                    &kernel_arg_5162};
        int64_t time_start_5160 = 0, time_end_5161 = 0;
        
        if (ctx->debugging) {
            fprintf(ctx->log, "Launching %s with grid size (",
                    "builtin#replicate_i8.replicate_4927");
            fprintf(ctx->log, "%ld", (long) num_groups_4931);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ") and block size (");
            fprintf(ctx->log, "%ld", (long) group_sizze_4930);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ").\n");
            time_start_5160 = get_wall_time();
        }
        
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda,
                                      &ctx->builtinzhreplicate_i8zireplicate_4927_runs,
                                      &ctx->builtinzhreplicate_i8zireplicate_4927_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->builtinzhreplicate_i8zireplicate_4927,
                                    grid[0], grid[1], grid[2], group_sizze_4930,
                                    1, 1, 0, NULL, kernel_args_5159, NULL));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_5161 = get_wall_time();
            fprintf(ctx->log, "Kernel %s runtime: %ldus\n",
                    "builtin#replicate_i8.replicate_4927", time_end_5161 -
                    time_start_5160);
        }
    }
    
  cleanup:
    { }
    return err;
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_5163,
                       struct memblock_device mat_inds_mem_4889,
                       struct memblock_device mat_vals_mem_4890,
                       struct memblock_device shp_mem_4891,
                       struct memblock_device vct_mem_4892, int64_t n_4772,
                       int64_t m_4773, int64_t implz2080U_4774)
{
    (void) ctx;
    
    int err = 0;
    struct memblock_device out_mem_4911;
    
    out_mem_4911.references = NULL;
    
    int64_t segscan_group_sizze_4815;
    
    segscan_group_sizze_4815 = ctx->sizes.mainzisegscan_group_sizze_4814;
    
    int64_t num_groups_4817;
    int32_t max_num_groups_4912;
    
    max_num_groups_4912 = ctx->sizes.mainzisegscan_num_groups_4816;
    num_groups_4817 = sext_i64_i32(smax64((int64_t) 1, smin64(sdiv_up64(m_4773,
                                                                        segscan_group_sizze_4815),
                                                              sext_i32_i64(max_num_groups_4912))));
    
    int64_t bytes_4894 = (int64_t) 8 * m_4773;
    struct memblock_device mem_4895;
    
    mem_4895.references = NULL;
    if (memblock_alloc_device(ctx, &mem_4895, bytes_4894, "mem_4895")) {
        err = 1;
        goto cleanup;
    }
    if (slt64((int64_t) 0, m_4773)) {
        if (ctx->debugging)
            fprintf(ctx->log, "%s\n", "\n# SegScan");
        
        int64_t numThreads_4913;
        
        numThreads_4913 = sdiv_up64(m_4773, segscan_group_sizze_4815 *
                                    (int64_t) 9) * segscan_group_sizze_4815;
        
        int64_t numGroups_4914;
        
        numGroups_4914 = sdiv_up64(m_4773, segscan_group_sizze_4815 *
                                   (int64_t) 9);
        
        struct memblock_device mainziid_counter_mem_4915 =
                               ctx->mainziid_counter_mem_4915;
        struct memblock_device status_flags_mem_4917;
        
        status_flags_mem_4917.references = NULL;
        if (memblock_alloc_device(ctx, &status_flags_mem_4917,
                                  (int64_t) sizeof(int8_t) * numGroups_4914,
                                  "status_flags_mem_4917")) {
            err = 1;
            goto cleanup;
        }
        
        struct memblock_device aggregates_mem_4919;
        
        aggregates_mem_4919.references = NULL;
        if (memblock_alloc_device(ctx, &aggregates_mem_4919,
                                  (int64_t) sizeof(int64_t) * numGroups_4914,
                                  "aggregates_mem_4919")) {
            err = 1;
            goto cleanup;
        }
        
        struct memblock_device incprefixes_mem_4921;
        
        incprefixes_mem_4921.references = NULL;
        if (memblock_alloc_device(ctx, &incprefixes_mem_4921,
                                  (int64_t) sizeof(int64_t) * numGroups_4914,
                                  "incprefixes_mem_4921")) {
            err = 1;
            goto cleanup;
        }
        if (futrts_builtinzhreplicate_i8(ctx, status_flags_mem_4917,
                                         numGroups_4914, (int8_t) 0) != 0) {
            err = 1;
            goto cleanup;
        }
        
        unsigned int shared_sizze_5168 = smax64(smax64((int64_t) 288,
                                                       (int64_t) 8 *
                                                       segscan_group_sizze_4815),
                                                (int64_t) 9 *
                                                segscan_group_sizze_4815 *
                                                (int64_t) 8);
        CUdeviceptr kernel_arg_5170 = shp_mem_4891.mem;
        CUdeviceptr kernel_arg_5171 = mem_4895.mem;
        CUdeviceptr kernel_arg_5172 = mainziid_counter_mem_4915.mem;
        CUdeviceptr kernel_arg_5173 = status_flags_mem_4917.mem;
        CUdeviceptr kernel_arg_5174 = aggregates_mem_4919.mem;
        CUdeviceptr kernel_arg_5175 = incprefixes_mem_4921.mem;
        unsigned int shared_offset_5169 = 0;
        
        if ((((((1 && sdiv_up64(m_4773, segscan_group_sizze_4815 *
                                (int64_t) 9) != 0) && 1 != 0) && 1 != 0) &&
              segscan_group_sizze_4815 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 >= 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 >= 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = sdiv_up64(m_4773, segscan_group_sizze_4815 *
                                      (int64_t) 9);
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_5165[] = {&ctx->global_failure,
                                        &shared_offset_5169, &m_4773,
                                        &kernel_arg_5170, &kernel_arg_5171,
                                        &kernel_arg_5172, &kernel_arg_5173,
                                        &kernel_arg_5174, &kernel_arg_5175};
            int64_t time_start_5166 = 0, time_end_5167 = 0;
            
            if (ctx->debugging) {
                fprintf(ctx->log, "Launching %s with grid size (",
                        "main.segscan_4820");
                fprintf(ctx->log, "%ld", (long) sdiv_up64(m_4773,
                                                          segscan_group_sizze_4815 *
                                                          (int64_t) 9));
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ") and block size (");
                fprintf(ctx->log, "%ld", (long) segscan_group_sizze_4815);
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ").\n");
                time_start_5166 = get_wall_time();
            }
            
            cudaEvent_t *pevents = NULL;
            
            if (ctx->profiling && !ctx->profiling_paused) {
                pevents = cuda_get_events(&ctx->cuda,
                                          &ctx->mainzisegscan_4820_runs,
                                          &ctx->mainzisegscan_4820_total_runtime);
                CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->mainzisegscan_4820, grid[0],
                                        grid[1], grid[2],
                                        segscan_group_sizze_4815, 1, 1, 0 +
                                        (shared_sizze_5168 + (8 -
                                                              shared_sizze_5168 %
                                                              8) % 8), NULL,
                                        kernel_args_5165, NULL));
            if (pevents != NULL)
                CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_5167 = get_wall_time();
                fprintf(ctx->log, "Kernel %s runtime: %ldus\n",
                        "main.segscan_4820", time_end_5167 - time_start_5166);
            }
        }
        if (memblock_unref_device(ctx, &incprefixes_mem_4921,
                                  "incprefixes_mem_4921") != 0)
            return 1;
        if (memblock_unref_device(ctx, &aggregates_mem_4919,
                                  "aggregates_mem_4919") != 0)
            return 1;
        if (memblock_unref_device(ctx, &status_flags_mem_4917,
                                  "status_flags_mem_4917") != 0)
            return 1;
    }
    
    struct memblock_device mem_4897;
    
    mem_4897.references = NULL;
    if (memblock_alloc_device(ctx, &mem_4897, n_4772, "mem_4897")) {
        err = 1;
        goto cleanup;
    }
    if (futrts_builtinzhreplicate_bool(ctx, mem_4897, n_4772, 0) != 0) {
        err = 1;
        goto cleanup;
    }
    
    int64_t segmap_group_sizze_4825;
    
    segmap_group_sizze_4825 = ctx->sizes.mainzisegmap_group_sizze_4824;
    
    int64_t segmap_usable_groups_4826 = sdiv_up64(m_4773,
                                                  segmap_group_sizze_4825);
    
    if (ctx->debugging)
        fprintf(ctx->log, "%s\n", "\n# SegMap");
    
    CUdeviceptr kernel_arg_5179 = mem_4895.mem;
    CUdeviceptr kernel_arg_5180 = mem_4897.mem;
    
    if ((((((1 && segmap_usable_groups_4826 != 0) && 1 != 0) && 1 != 0) &&
          segmap_group_sizze_4825 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 >= 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 >= 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = segmap_usable_groups_4826;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_5176[] = {&ctx->global_failure, &n_4772, &m_4773,
                                    &kernel_arg_5179, &kernel_arg_5180};
        int64_t time_start_5177 = 0, time_end_5178 = 0;
        
        if (ctx->debugging) {
            fprintf(ctx->log, "Launching %s with grid size (",
                    "main.segmap_4822");
            fprintf(ctx->log, "%ld", (long) segmap_usable_groups_4826);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ") and block size (");
            fprintf(ctx->log, "%ld", (long) segmap_group_sizze_4825);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ").\n");
            time_start_5177 = get_wall_time();
        }
        
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->mainzisegmap_4822_runs,
                                      &ctx->mainzisegmap_4822_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->mainzisegmap_4822, grid[0], grid[1],
                                    grid[2], segmap_group_sizze_4825, 1, 1, 0,
                                    NULL, kernel_args_5176, NULL));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_5178 = get_wall_time();
            fprintf(ctx->log, "Kernel %s runtime: %ldus\n", "main.segmap_4822",
                    time_end_5178 - time_start_5177);
        }
    }
    
    int64_t segscan_group_sizze_4829;
    
    segscan_group_sizze_4829 = ctx->sizes.mainzisegscan_group_sizze_4828;
    
    int64_t num_groups_4831;
    int32_t max_num_groups_5017;
    
    max_num_groups_5017 = ctx->sizes.mainzisegscan_num_groups_4830;
    num_groups_4831 = sext_i64_i32(smax64((int64_t) 1, smin64(sdiv_up64(n_4772,
                                                                        segscan_group_sizze_4829),
                                                              sext_i32_i64(max_num_groups_5017))));
    
    struct memblock_device mem_4901;
    
    mem_4901.references = NULL;
    if (memblock_alloc_device(ctx, &mem_4901, n_4772, "mem_4901")) {
        err = 1;
        goto cleanup;
    }
    
    int64_t bytes_4902 = (int64_t) 4 * n_4772;
    struct memblock_device mem_4903;
    
    mem_4903.references = NULL;
    if (memblock_alloc_device(ctx, &mem_4903, bytes_4902, "mem_4903")) {
        err = 1;
        goto cleanup;
    }
    if (slt64((int64_t) 0, n_4772)) {
        if (ctx->debugging)
            fprintf(ctx->log, "%s\n", "\n# SegScan");
        
        int64_t numThreads_5018;
        
        numThreads_5018 = sdiv_up64(n_4772, segscan_group_sizze_4829 *
                                    (int64_t) 9) * segscan_group_sizze_4829;
        
        int64_t numGroups_5019;
        
        numGroups_5019 = sdiv_up64(n_4772, segscan_group_sizze_4829 *
                                   (int64_t) 9);
        
        struct memblock_device mainziid_counter_mem_5020 =
                               ctx->mainziid_counter_mem_5020;
        struct memblock_device status_flags_mem_5022;
        
        status_flags_mem_5022.references = NULL;
        if (memblock_alloc_device(ctx, &status_flags_mem_5022,
                                  (int64_t) sizeof(int8_t) * numGroups_5019,
                                  "status_flags_mem_5022")) {
            err = 1;
            goto cleanup;
        }
        
        struct memblock_device aggregates_mem_5024;
        
        aggregates_mem_5024.references = NULL;
        if (memblock_alloc_device(ctx, &aggregates_mem_5024,
                                  (int64_t) sizeof(bool) * numGroups_5019,
                                  "aggregates_mem_5024")) {
            err = 1;
            goto cleanup;
        }
        
        struct memblock_device incprefixes_mem_5026;
        
        incprefixes_mem_5026.references = NULL;
        if (memblock_alloc_device(ctx, &incprefixes_mem_5026,
                                  (int64_t) sizeof(bool) * numGroups_5019,
                                  "incprefixes_mem_5026")) {
            err = 1;
            goto cleanup;
        }
        
        struct memblock_device aggregates_mem_5028;
        
        aggregates_mem_5028.references = NULL;
        if (memblock_alloc_device(ctx, &aggregates_mem_5028,
                                  (int64_t) sizeof(float) * numGroups_5019,
                                  "aggregates_mem_5028")) {
            err = 1;
            goto cleanup;
        }
        
        struct memblock_device incprefixes_mem_5030;
        
        incprefixes_mem_5030.references = NULL;
        if (memblock_alloc_device(ctx, &incprefixes_mem_5030,
                                  (int64_t) sizeof(float) * numGroups_5019,
                                  "incprefixes_mem_5030")) {
            err = 1;
            goto cleanup;
        }
        if (futrts_builtinzhreplicate_i8(ctx, status_flags_mem_5022,
                                         numGroups_5019, (int8_t) 0) != 0) {
            err = 1;
            goto cleanup;
        }
        
        unsigned int shared_sizze_5185 = smax64(smax64((int64_t) 192,
                                                       sdiv_up64(segscan_group_sizze_4829,
                                                                 (int64_t) 4) *
                                                       (int64_t) 4 +
                                                       (int64_t) 4 *
                                                       segscan_group_sizze_4829),
                                                smax64((int64_t) 9 *
                                                       segscan_group_sizze_4829,
                                                       (int64_t) 9 *
                                                       segscan_group_sizze_4829 *
                                                       (int64_t) 4));
        CUdeviceptr kernel_arg_5187 = mat_inds_mem_4889.mem;
        CUdeviceptr kernel_arg_5188 = mat_vals_mem_4890.mem;
        CUdeviceptr kernel_arg_5189 = vct_mem_4892.mem;
        CUdeviceptr kernel_arg_5190 = mem_4897.mem;
        CUdeviceptr kernel_arg_5191 = mem_4901.mem;
        CUdeviceptr kernel_arg_5192 = mem_4903.mem;
        CUdeviceptr kernel_arg_5193 = mainziid_counter_mem_5020.mem;
        CUdeviceptr kernel_arg_5194 = status_flags_mem_5022.mem;
        CUdeviceptr kernel_arg_5195 = aggregates_mem_5024.mem;
        CUdeviceptr kernel_arg_5196 = incprefixes_mem_5026.mem;
        CUdeviceptr kernel_arg_5197 = aggregates_mem_5028.mem;
        CUdeviceptr kernel_arg_5198 = incprefixes_mem_5030.mem;
        unsigned int shared_offset_5186 = 0;
        
        if ((((((1 && sdiv_up64(n_4772, segscan_group_sizze_4829 *
                                (int64_t) 9) != 0) && 1 != 0) && 1 != 0) &&
              segscan_group_sizze_4829 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 >= 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 >= 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = sdiv_up64(n_4772, segscan_group_sizze_4829 *
                                      (int64_t) 9);
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_5182[] = {&ctx->global_failure,
                                        &ctx->failure_is_an_option,
                                        &ctx->global_failure_args,
                                        &shared_offset_5186, &n_4772,
                                        &implz2080U_4774, &kernel_arg_5187,
                                        &kernel_arg_5188, &kernel_arg_5189,
                                        &kernel_arg_5190, &kernel_arg_5191,
                                        &kernel_arg_5192, &kernel_arg_5193,
                                        &kernel_arg_5194, &kernel_arg_5195,
                                        &kernel_arg_5196, &kernel_arg_5197,
                                        &kernel_arg_5198};
            int64_t time_start_5183 = 0, time_end_5184 = 0;
            
            if (ctx->debugging) {
                fprintf(ctx->log, "Launching %s with grid size (",
                        "main.segscan_4834");
                fprintf(ctx->log, "%ld", (long) sdiv_up64(n_4772,
                                                          segscan_group_sizze_4829 *
                                                          (int64_t) 9));
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ") and block size (");
                fprintf(ctx->log, "%ld", (long) segscan_group_sizze_4829);
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ", ");
                fprintf(ctx->log, "%ld", (long) 1);
                fprintf(ctx->log, ").\n");
                time_start_5183 = get_wall_time();
            }
            
            cudaEvent_t *pevents = NULL;
            
            if (ctx->profiling && !ctx->profiling_paused) {
                pevents = cuda_get_events(&ctx->cuda,
                                          &ctx->mainzisegscan_4834_runs,
                                          &ctx->mainzisegscan_4834_total_runtime);
                CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->mainzisegscan_4834, grid[0],
                                        grid[1], grid[2],
                                        segscan_group_sizze_4829, 1, 1, 0 +
                                        (shared_sizze_5185 + (8 -
                                                              shared_sizze_5185 %
                                                              8) % 8), NULL,
                                        kernel_args_5182, NULL));
            if (pevents != NULL)
                CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_5184 = get_wall_time();
                fprintf(ctx->log, "Kernel %s runtime: %ldus\n",
                        "main.segscan_4834", time_end_5184 - time_start_5183);
            }
        }
        ctx->failure_is_an_option = 1;
        if (memblock_unref_device(ctx, &incprefixes_mem_5030,
                                  "incprefixes_mem_5030") != 0)
            return 1;
        if (memblock_unref_device(ctx, &aggregates_mem_5028,
                                  "aggregates_mem_5028") != 0)
            return 1;
        if (memblock_unref_device(ctx, &incprefixes_mem_5026,
                                  "incprefixes_mem_5026") != 0)
            return 1;
        if (memblock_unref_device(ctx, &aggregates_mem_5024,
                                  "aggregates_mem_5024") != 0)
            return 1;
        if (memblock_unref_device(ctx, &status_flags_mem_5022,
                                  "status_flags_mem_5022") != 0)
            return 1;
    }
    if (memblock_unref_device(ctx, &mem_4897, "mem_4897") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4901, "mem_4901") != 0)
        return 1;
    
    int64_t segmap_group_sizze_4875;
    
    segmap_group_sizze_4875 = ctx->sizes.mainzisegmap_group_sizze_4864;
    
    int64_t segmap_usable_groups_4876 = sdiv_up64(m_4773,
                                                  segmap_group_sizze_4875);
    int64_t bytes_4905 = (int64_t) 4 * m_4773;
    struct memblock_device mem_4906;
    
    mem_4906.references = NULL;
    if (memblock_alloc_device(ctx, &mem_4906, bytes_4905, "mem_4906")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(ctx->log, "%s\n", "\n# SegMap");
    
    CUdeviceptr kernel_arg_5202 = mem_4895.mem;
    CUdeviceptr kernel_arg_5203 = mem_4903.mem;
    CUdeviceptr kernel_arg_5204 = mem_4906.mem;
    
    if ((((((1 && segmap_usable_groups_4876 != 0) && 1 != 0) && 1 != 0) &&
          segmap_group_sizze_4875 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 >= 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 >= 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = segmap_usable_groups_4876;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_5199[] = {&ctx->global_failure,
                                    &ctx->failure_is_an_option,
                                    &ctx->global_failure_args, &n_4772, &m_4773,
                                    &kernel_arg_5202, &kernel_arg_5203,
                                    &kernel_arg_5204};
        int64_t time_start_5200 = 0, time_end_5201 = 0;
        
        if (ctx->debugging) {
            fprintf(ctx->log, "Launching %s with grid size (",
                    "main.segmap_4862");
            fprintf(ctx->log, "%ld", (long) segmap_usable_groups_4876);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ") and block size (");
            fprintf(ctx->log, "%ld", (long) segmap_group_sizze_4875);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ", ");
            fprintf(ctx->log, "%ld", (long) 1);
            fprintf(ctx->log, ").\n");
            time_start_5200 = get_wall_time();
        }
        
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->mainzisegmap_4862_runs,
                                      &ctx->mainzisegmap_4862_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->mainzisegmap_4862, grid[0], grid[1],
                                    grid[2], segmap_group_sizze_4875, 1, 1, 0,
                                    NULL, kernel_args_5199, NULL));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_5201 = get_wall_time();
            fprintf(ctx->log, "Kernel %s runtime: %ldus\n", "main.segmap_4862",
                    time_end_5201 - time_start_5200);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_4895, "mem_4895") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4903, "mem_4903") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_4911, &mem_4906, "mem_4906") != 0)
        return 1;
    (*out_mem_p_5163).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_5163, &out_mem_4911,
                            "out_mem_4911") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4906, "mem_4906") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4903, "mem_4903") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4901, "mem_4901") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4897, "mem_4897") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_4895, "mem_4895") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_4911, "out_mem_4911") != 0)
        return 1;
    
  cleanup:
    { }
    return err;
}
struct futhark_f32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    {
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->copy_host_to_dev_runs,
                                      &ctx->copy_host_to_dev_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, (size_t) dim0 *
                                  sizeof(float)));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    {
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->copy_dev_to_dev_runs,
                                      &ctx->copy_dev_to_dev_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, (size_t) dim0 *
                              sizeof(float)));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    {
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->copy_dev_to_host_runs,
                                      &ctx->copy_dev_to_host_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0,
                                  (size_t) arr->shape[0] * sizeof(float)));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                      struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                                    struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_i64_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const
                                          int64_t *data, int64_t dim0)
{
    struct futhark_i64_1d *bad = NULL;
    struct futhark_i64_1d *arr =
                          (struct futhark_i64_1d *) malloc(sizeof(struct futhark_i64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(int64_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    {
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->copy_host_to_dev_runs,
                                      &ctx->copy_host_to_dev_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, (size_t) dim0 *
                                  sizeof(int64_t)));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, const
                                              CUdeviceptr data, int offset,
                                              int64_t dim0)
{
    struct futhark_i64_1d *bad = NULL;
    struct futhark_i64_1d *arr =
                          (struct futhark_i64_1d *) malloc(sizeof(struct futhark_i64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(int64_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    {
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->copy_dev_to_dev_runs,
                                      &ctx->copy_dev_to_dev_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, (size_t) dim0 *
                              sizeof(int64_t)));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i64_1d(struct futhark_context *ctx,
                          struct futhark_i64_1d *arr, int64_t *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    {
        cudaEvent_t *pevents = NULL;
        
        if (ctx->profiling && !ctx->profiling_paused) {
            pevents = cuda_get_events(&ctx->cuda, &ctx->copy_dev_to_host_runs,
                                      &ctx->copy_dev_to_host_total_runtime);
            CUDA_SUCCEED(cudaEventRecord(pevents[0], 0));
        }
        CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0,
                                  (size_t) arr->shape[0] * sizeof(int64_t)));
        if (pevents != NULL)
            CUDA_SUCCEED(cudaEventRecord(pevents[1], 0));
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_i64_1d(struct futhark_context *ctx,
                                      struct futhark_i64_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx,
                                    struct futhark_i64_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const
                       struct futhark_i64_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_i64_1d *in2, const
                       struct futhark_f32_1d *in3)
{
    struct memblock_device mat_inds_mem_4889;
    
    mat_inds_mem_4889.references = NULL;
    
    struct memblock_device mat_vals_mem_4890;
    
    mat_vals_mem_4890.references = NULL;
    
    struct memblock_device shp_mem_4891;
    
    shp_mem_4891.references = NULL;
    
    struct memblock_device vct_mem_4892;
    
    vct_mem_4892.references = NULL;
    
    int64_t n_4772;
    int64_t m_4773;
    int64_t implz2080U_4774;
    struct memblock_device out_mem_4911;
    
    out_mem_4911.references = NULL;
    
    int ret = 0;
    
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuCtxPushCurrent(ctx->cuda.cu_ctx));
    mat_inds_mem_4889 = in0->mem;
    n_4772 = in0->shape[0];
    mat_vals_mem_4890 = in1->mem;
    n_4772 = in1->shape[0];
    shp_mem_4891 = in2->mem;
    m_4773 = in2->shape[0];
    vct_mem_4892 = in3->mem;
    implz2080U_4774 = in3->shape[0];
    if (!(n_4772 == in0->shape[0] && (n_4772 == in1->shape[0] && (m_4773 ==
                                                                  in2->shape[0] &&
                                                                  implz2080U_4774 ==
                                                                  in3->shape[0])))) {
        ret = 1;
        if (!ctx->error)
            ctx->error =
                msgprintf("Error: entry point arguments have invalid sizes.\n");
    } else {
        ret = futrts_main(ctx, &out_mem_4911, mat_inds_mem_4889,
                          mat_vals_mem_4890, shp_mem_4891, vct_mem_4892, n_4772,
                          m_4773, implz2080U_4774);
        if (ret == 0) {
            assert((*out0 =
                    (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
                NULL);
            (*out0)->mem = out_mem_4911;
            (*out0)->shape[0] = m_4773;
        }
    }
    CUDA_SUCCEED(cuCtxPopCurrent(&ctx->cuda.cu_ctx));
    lock_unlock(&ctx->lock);
    return ret;
}
