#@title 9 · Benchmark all sizes on this GPU { display-mode: "form" }
RUN_BENCHMARK = False  #@param {type:"boolean"}
BENCHMARK_SIZES = "1M,2M,5M,10M,50M,115M"  #@param {type:"string"}
if RUN_BENCHMARK:
    BENCH = tg.benchmark_presets([s.strip() for s in BENCHMARK_SIZES.split(",") if s.strip()], device="auto",
                                 **({"seq_len": 128} if DEVICE == "cpu" else {}))
else:
    print("Set RUN_BENCHMARK = True to time every preset.")
