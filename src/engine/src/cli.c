#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "inference.h"
#include "tokenizer.h"
#include "model_io.h"

extern trnm_model_t* trnm_load(const char* path);
extern void trnm_free(trnm_model_t* model);

#define MAX_PROMPT_TOKENS 4096
#define MAX_INPUT_LEN     8192
#define CLI_VERSION       "1.0"

typedef struct {
    const char* model_path;
    const char* tokenizer_path;
    const char* prompt;
    int32_t max_tokens;
    float temperature;
    int32_t top_k;
    float top_p;
    uint64_t seed;
    int greedy;
    int no_stream;
    int no_stats;
    int quiet;
    int interactive;
    int show_help;
} cli_args_t;

static volatile int g_interrupted = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

#ifdef _WIN32
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

static cli_args_t parse_args(int argc, char* argv[]) {
    cli_args_t args = {
        .model_path = NULL,
        .tokenizer_path = NULL,
        .prompt = NULL,
        .max_tokens = 256,
        .temperature = 1.0f,
        .top_k = 40,
        .top_p = 0.9f,
        .seed = 42,
        .greedy = 0,
        .no_stream = 0,
        .no_stats = 0,
        .quiet = 0,
        .interactive = 0,
        .show_help = 0
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            args.temperature = (float)strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            args.top_k = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            args.top_p = (float)strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = (uint64_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            args.greedy = 1;
        } else if (strcmp(argv[i], "--no-stream") == 0) {
            args.no_stream = 1;
        } else if (strcmp(argv[i], "--no-stats") == 0) {
            args.no_stats = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            args.quiet = 1;
        } else if (strcmp(argv[i], "--interactive") == 0) {
            args.interactive = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            args.show_help = 1;
        }
    }

    return args;
}

static void print_usage(void) {
    printf("trnm_cli — Ternary SSM Inference CLI\n\n");
    printf("Usage: trnm_cli --model <path> --tokenizer <path> [OPTIONS]\n\n");
    printf("Required:\n");
    printf("  --model <path>        Path to .trnm model file\n");
    printf("  --tokenizer <path>    Path to .tvoc tokenizer file\n\n");
    printf("Generation:\n");
    printf("  --prompt <text>       Input prompt (omit for interactive mode)\n");
    printf("  --max-tokens <n>      Max tokens to generate (default: 256)\n");
    printf("  --temperature <f>     Temperature (default: 1.0)\n");
    printf("  --top-k <n>           Top-K (default: 40, 0=disabled)\n");
    printf("  --top-p <f>           Top-P (default: 0.9, 1.0=disabled)\n");
    printf("  --seed <n>            RNG seed (default: 42)\n");
    printf("  --greedy              Greedy decoding\n\n");
    printf("Output:\n");
    printf("  --no-stream           Print all at once\n");
    printf("  --no-stats            Skip performance stats\n");
    printf("  --quiet               Minimal output\n\n");
    printf("Modes:\n");
    printf("  --interactive         Force interactive mode\n");
    printf("  -h, --help            Show help\n");
}

static void print_banner(const trnm_model_t* model, const tokenizer_t* tok) {
    if (!model || !tok) return;

    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║  Ternary SSM Inference Engine (v%s) ║\n", CLI_VERSION);
    printf("╚════════════════════════════════════════╝\n");
    printf("Model Config:\n");
    printf("  d_model:      %u\n", model->header.d_model);
    printf("  n_layers:     %u\n", model->header.n_layers);
    printf("  vocab_size:   %u\n", model->header.vocab_size);
    printf("  d_state:      %u\n", model->header.d_state);
    printf("  d_conv:       %u\n", model->header.d_conv);
    printf("  expand:       %u\n", model->header.expand);
    printf("  Ternary:      %s\n", (model->header.flags & TRNM_FLAG_TERNARY) ? "yes" : "no");
    printf("  SSD:          %s\n", (model->header.flags & TRNM_FLAG_SSD) ? "yes" : "no");
    printf("\nTokenizer:\n");
    printf("  vocab_size:   %d\n", tok->vocab_size);
    printf("  special:      pad=%d unk=%d bos=%d eos=%d\n",
           tok->pad_id, tok->unk_id, tok->bos_id, tok->eos_id);
    printf("\n");
}

static int run_single(inference_engine_t* engine, const tokenizer_t* tok,
                      const char* prompt, const cli_args_t* args) {
    if (!prompt || !engine || !tok) return -1;

    int32_t prompt_tokens[MAX_PROMPT_TOKENS];
    int32_t n_prompt = tokenizer_encode(tok, prompt, prompt_tokens, MAX_PROMPT_TOKENS);

    if (n_prompt <= 0) {
        fprintf(stderr, "Error: failed to encode prompt\n");
        return -1;
    }

    if (!args->quiet) {
        printf("Prompt: %s\n", prompt);
        printf("Tokens: %d\n\n", n_prompt);
    }

    double time_prefill_start = get_time_ms();
    int prefill_result = engine_prefill(engine, prompt_tokens, n_prompt);
    double time_prefill_end = get_time_ms();
    double time_prefill = time_prefill_end - time_prefill_start;

    if (prefill_result != ENGINE_OK) {
        fprintf(stderr, "Error: prefill failed\n");
        return -1;
    }

    if (!args->quiet) {
        printf("Generated: ");
        fflush(stdout);
    }

    double time_gen_start = get_time_ms();
    int32_t n_generated = 0;
    char token_str[256];

    for (int i = 0; i < args->max_tokens && !g_interrupted; i++) {
        int32_t next_token = engine_step(engine, 0);

        if (next_token == ENGINE_ERR_TOKEN || next_token == ENGINE_ERR_NULL) {
            break;
        }

        if (next_token == (int32_t)tok->eos_id) {
            break;
        }

        int32_t n_str = tokenizer_decode_token(tok, next_token, token_str, sizeof(token_str));
        if (n_str > 0) {
            if (!args->no_stream) {
                fwrite(token_str, 1, n_str, stdout);
                fflush(stdout);
            }
            n_generated++;
        }
    }

    double time_gen_end = get_time_ms();
    double time_gen = time_gen_end - time_gen_start;

    if (!args->quiet) {
        printf("\n\n");
    }

    if (!args->no_stats) {
        double prefill_tok_s = (n_prompt > 0) ? (n_prompt * 1000.0 / time_prefill) : 0.0;
        double gen_tok_s = (n_generated > 0) ? (n_generated * 1000.0 / time_gen) : 0.0;
        double total_time = (time_prefill + time_gen) / 1000.0;

        printf("\n=== Performance ===\n");
        printf("Prefill:     %.1f tok/s (%.0f ms)\n", prefill_tok_s, time_prefill);
        printf("Generation:  %.1f tok/s (%.0f ms, %d tokens)\n", gen_tok_s, time_gen, n_generated);
        printf("Total time:  %.2f s\n", total_time);
    }

    return 0;
}

static int run_interactive(inference_engine_t* engine, const tokenizer_t* tok,
                          const cli_args_t* args) {
    if (!engine || !tok) return -1;

    inference_config_t config = engine_default_config();
    config.temperature = args->temperature;
    config.top_k = args->top_k;
    config.top_p = args->top_p;
    config.seed = args->seed;
    config.max_tokens = args->max_tokens;

    if (args->greedy) {
        config.sample_mode = ENGINE_SAMPLE_GREEDY;
    }

    engine_set_config(engine, &config);

    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║     Interactive Mode - Commands:       ║\n");
    printf("║  /reset      Reset context             ║\n");
    printf("║  /stats      Show stats                ║\n");
    printf("║  /config     Show config               ║\n");
    printf("║  /set key val  Set parameter           ║\n");
    printf("║  /quit       Exit                      ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    char input_line[MAX_INPUT_LEN];

    while (!g_interrupted) {
        printf("> ");
        fflush(stdout);

        if (!fgets(input_line, sizeof(input_line), stdin)) {
            break;
        }

        size_t len = strlen(input_line);
        if (len > 0 && input_line[len - 1] == '\n') {
            input_line[len - 1] = '\0';
        }

        if (strlen(input_line) == 0) {
            continue;
        }

        if (strcmp(input_line, "/reset") == 0) {
            engine_reset(engine);
            printf("Context reset.\n");
        } else if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
            break;
        } else if (strcmp(input_line, "/config") == 0) {
            printf("Config:\n");
            printf("  temperature: %.2f\n", config.temperature);
            printf("  top_k:       %d\n", config.top_k);
            printf("  top_p:       %.2f\n", config.top_p);
            printf("  max_tokens:  %d\n", config.max_tokens);
            printf("  seed:        %lu\n", (unsigned long)config.seed);
        } else if (strncmp(input_line, "/set ", 5) == 0) {
            char key[64], value[64];
            int n = sscanf(input_line + 5, "%63s %63s", key, value);
            if (n == 2) {
                if (strcmp(key, "temperature") == 0) {
                    config.temperature = (float)strtod(value, NULL);
                    engine_set_config(engine, &config);
                } else if (strcmp(key, "top_k") == 0) {
                    config.top_k = (int32_t)strtol(value, NULL, 10);
                    engine_set_config(engine, &config);
                } else if (strcmp(key, "top_p") == 0) {
                    config.top_p = (float)strtod(value, NULL);
                    engine_set_config(engine, &config);
                } else if (strcmp(key, "max_tokens") == 0) {
                    config.max_tokens = (int32_t)strtol(value, NULL, 10);
                    engine_set_config(engine, &config);
                } else if (strcmp(key, "seed") == 0) {
                    config.seed = (uint64_t)strtoul(value, NULL, 10);
                    engine_set_config(engine, &config);
                } else if (strcmp(key, "greedy") == 0) {
                    int greedy_val = (int)strtol(value, NULL, 10);
                    config.sample_mode = greedy_val ? ENGINE_SAMPLE_GREEDY : ENGINE_SAMPLE_TEMPERATURE;
                    engine_set_config(engine, &config);
                } else {
                    printf("Unknown key: %s\n", key);
                }
            }
        } else if (strcmp(input_line, "/stats") == 0) {
            printf("Engine stats:\n");
            printf("  tokens_processed: %d\n", engine->tokens_processed);
        } else {
            cli_args_t temp_args = *args;
            temp_args.prompt = input_line;
            run_single(engine, tok, input_line, &temp_args);
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    cli_args_t args = parse_args(argc, argv);

    if (args.show_help) {
        print_usage();
        return 0;
    }

    if (!args.model_path || !args.tokenizer_path) {
        fprintf(stderr, "Error: --model and --tokenizer are required\n");
        fprintf(stderr, "Use -h or --help for usage\n");
        return 1;
    }

    trnm_model_t* model = trnm_load(args.model_path);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from %s\n", args.model_path);
        return 1;
    }

    tokenizer_t* tok = tokenizer_load(args.tokenizer_path);
    if (!tok) {
        fprintf(stderr, "Error: failed to load tokenizer from %s\n", args.tokenizer_path);
        trnm_free(model);
        return 1;
    }

    engine_init();
    inference_engine_t* engine = engine_create(model);
    if (!engine) {
        fprintf(stderr, "Error: failed to create inference engine\n");
        tokenizer_free(tok);
        trnm_free(model);
        return 1;
    }

    inference_config_t config = engine_default_config();
    config.temperature = args.temperature;
    config.top_k = args.top_k;
    config.top_p = args.top_p;
    config.seed = args.seed;
    config.max_tokens = args.max_tokens;

    if (args.greedy) {
        config.sample_mode = ENGINE_SAMPLE_GREEDY;
    }

    engine_set_config(engine, &config);

    signal(SIGINT, signal_handler);

    if (!args.quiet) {
        print_banner(model, tok);
    }

    int result = 0;

    if (args.prompt) {
        result = run_single(engine, tok, args.prompt, &args);
    } else {
        args.interactive = 1;
        result = run_interactive(engine, tok, &args);
    }

    engine_free(engine);
    tokenizer_free(tok);
    trnm_free(model);

    return result;
}
