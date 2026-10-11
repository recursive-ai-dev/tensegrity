#@title 1 · Your run { display-mode: "form" }
MODEL_SIZE = "5M"  #@param ["1M", "2M", "5M", "10M", "50M", "115M"]
TRAIN_HOURS = 2.0  #@param {type:"number"}
DATASET = "tinystories"  #@param ["tinystories", "tinystories-valid", "shakespeare", "custom"]
DATA_MEGABYTES = 400  #@param {type:"slider", min:50, max:1900, step:50}
CUSTOM_DATA_PATH = ""  #@param {type:"string"}
CUSTOM_DOC_SEPARATOR = ""  #@param {type:"string"}
OPTIMIZER = "muon"  #@param ["muon", "adamw"]
INIT_FROM_WEIGHTS = ""  #@param {type:"string"}
USE_GOOGLE_DRIVE = True  #@param {type:"boolean"}
RUN_NAME = ""  #@param {type:"string"}
START_FRESH = False  #@param {type:"boolean"}
STOP_AFTER_MINUTES = 0  #@param {type:"number"}
SMOKE_TEST = False  #@param {type:"boolean"}

# SMOKE_TEST: run the entire notebook in a couple of minutes (tiny budget, short context) to
# check your setup before committing hours to a real run.
if SMOKE_TEST:
    MODEL_SIZE, DATASET, DATA_MEGABYTES, TRAIN_HOURS = "1M", "shakespeare", 2, 0.01
    print("SMOKE_TEST: 1M model, Shakespeare, ~1 minute of training.")
