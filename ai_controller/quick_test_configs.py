# Create quick_test_configs.py
from sim_config_generator import CurriculumGenerator

gen = CurriculumGenerator(seed=42)

# Generate 3 configs (easy → medium → hard)
configs = [
    gen.generate_easy_config(),
    gen.generate_medium_config(),
    gen.generate_hard_config()
]

# Save them
for i, config in enumerate(configs):
    config.sim_time = 60.0  # Short simulation (1 minute)
    config.save(f"test_config_{i}.json")
    print(f"✓ Saved test_config_{i}.json ({config.difficulty_level})")