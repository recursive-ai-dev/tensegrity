#@title 7 · Interactive Chat { display-mode: "form" }
SYSTEM_PROMPT = "You are a helpful, concise assistant."  #@param {type:"string"}
TEMPERATURE = 0.7  #@param {type:"slider", min:0.0, max:1.5, step:0.05}
TOP_K = 40  #@param {type:"integer"}
TOP_P = None  #@param {type:"number"}
MIN_P = 0.05  #@param {type:"slider", min:0.0, max:0.3, step:0.01}
REP_PENALTY = 1.12  #@param {type:"slider", min:1.0, max:1.5, step:0.01}
FREQ_PENALTY = 0.0  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
PRES_PENALTY = 0.0  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
USE_CONF_TEMP = True  #@param {type:"boolean"}
MAX_HISTORY_TOKENS = 2048  #@param {type:"integer"}

session = tg.ChatSession(BEST, system_prompt=SYSTEM_PROMPT, max_history_tokens=MAX_HISTORY_TOKENS,
                          sampling_kwargs=dict(temperature=TEMPERATURE, top_k=TOP_K or None, top_p=TOP_P,
                                               min_p=MIN_P or None, repetition_penalty=REP_PENALTY,
                                               frequency_penalty=FREQ_PENALTY, presence_penalty=PRES_PENALTY,
                                               use_confidence_temp=USE_CONF_TEMP))

print(f"Chat started. System prompt: {SYSTEM_PROMPT or '(none)'}")
print("Commands: /reset (clear history), /stats (cache info), /quit (exit)")
print("-" * 60)

while True:
    try:
        user = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user == "/quit":
        break
    if user == "/reset":
        session.reset()
        print("[chat] History cleared (system prompt retained)")
        continue
    if user == "/stats":
        s = session.get_stats()
        print(f"[chat] Cached: {s['cached_tokens']} tokens, System: {s['system_tokens']}, Turns: {s['turn_count']}, KV: {s['kv_memory_mb']:.1f} MB")
        continue

    print("Assistant: ", end="", flush=True)
    for chunk in session.stream(user):
        print(chunk, end="", flush=True)
    print()