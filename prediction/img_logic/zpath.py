import os
from prediction.params import *

cwd = os.getcwd()
print("os.getcwd()", os.getcwd())
# test_path = os.path.join(
#     os.getcwd(), "..", "..", "..", "data-context-and-setup/data/csv"
# )

print("root_dir", os.path.dirname(os.path.dirname(__file__)))

parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
print("parent_dir", parent_dir)

app_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
print("app_dir", app_dir)

model_path = os.path.join(
    app_dir, "raw_data", "computer_vision", "models", CNN_TRAINED_MODEL
)
print("model_path", model_path)
