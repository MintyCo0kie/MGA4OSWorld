python run.py  \
--provider_name docker   \
--path_to_vm /path/to/Ubuntu.qcow2\
--headless     \
--observation_type screenshot    \
--model MGA   \
--sleep_after_execution 5   \
--max_steps 50   \
--result_dir ./results/1 \
--test_all_meta_path evaluation_examples/test_nogdrive.json \
--config_path /MGA/mm_agents/config/config.yaml


