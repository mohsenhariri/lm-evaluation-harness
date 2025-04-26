# include *.make

COMMON_LIB := /mnt/rds/VipinRDS/VipinRDS/users/mxh1029/common
KVQ_LIB := ./kvq
PYTHONPATH := $(COMMON_LIB):$(KVQ_LIB):$(PYTHONPATH)



LLAMA3_1B_HF := meta-llama/Llama-3.2-1B
LLAMA3_1B_NAME := $(word 2,$(subst /, ,$(LLAMA3_1B_HF)))
LLAMA3_1B_REPO  := /mnt/rds/VipinRDS/VipinRDS/users/mxh1029/llms/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6

TASK := gsm8k

base:
	lm_eval --model hf \
	--model_args pretrained=$(LLAMA3_1B_REPO),dtype=bfloat16 \
	--tasks $(TASK) \
	--seed 777 \
	--batch_size auto \
	--log_samples \
	--output_path output > logs/gsm8k.log 2>&1


kvq:
	lm_eval --model hf \
	--model_args pretrained=$(LLAMA3_1B_REPO),dtype=bfloat16 \
	--tasks $(TASK) \
	--seed 777 \
	--kvq nbits=4,axis_key=0,axis_value=0,q_group_size=64,residual_length=128 \
	--batch_size auto \
	--log_samples \
	--output_path output > logs/gsm8k_kvq.log 2>&1



test_kvq: 
	python test_kvq.py