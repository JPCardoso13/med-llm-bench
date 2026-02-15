from typing import Dict, Optional
from pydantic import BaseModel, field_validator, model_validator, ValidationInfo, ValidationError

class MCQSample(BaseModel):
    """
    A unified representation of a Multiple Choice Question (MCQ) for benchmarking.

    This class acts as a contract for data ingestion. It ensures that data from 
    various sources (CSV, JSON, SQL) is normalized into a consistent structure 
    before entering the evaluation pipeline.

    Attributes:
        id (str): A unique identifier for the sample. 
        question (str): The question the model should answer.
        options (Dict[str, str]): A dictionary mapping option keys to text.
                                  Example: {'A': 'Option 1 text', 'B': 'Option 2 text'}
        answer_idx (str): The key corresponding to the correct option.
        source (Optional[str]): The origin dataset name (e.g., "medqa", "mmlu", "custom_test").
                                Defaults to None.
        category (Optional[str]): The specific domain, topic, or sub-field. 
                                  Example: "Cardiology", "Physics". Defaults to None.
        context (Optional[str]): Reference text if the task is context-dependent
                                 (e.g., a reading comprehension task). Defaults to None.
    
    Raises:
        ValidationError: If any of the constraints are violated, a detailed error message will indicate the issue.
    """
    id: str
    question: str
    options: Dict[str, str]
    answer_idx: str 
    source: Optional[str] = None
    category: Optional[str] = None
    context: Optional[str] = None

    # Constraint 1: Mandatory string fields must not be empty or whitespace
    @field_validator('id', 'question', 'answer_idx')
    @classmethod
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return v

    # Constraint 2: Options dictionary must have at least two entries
    @field_validator('options')
    @classmethod
    def check_options_length(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) < 2:
            raise ValueError("The 'options' dictionary must contain at least two entries.")
        return v

    # Constraint 3: answer_idx must be a key in options
    @model_validator(mode='after')
    def check_answer_in_options(self) -> 'MCQSample':
        if self.answer_idx not in self.options:
            raise ValueError(f"Field 'answer_idx' ('{self.answer_idx}') must be in the 'options' dictionary.")
        return self

if __name__ == "__main__":
    def run_test(name: str, data: dict, should_pass: bool):
        print(f"--- Test: {name} ---")
        try:
            obj = MCQSample(**data)

            if should_pass:
                print(f"SUCCESS: Data validated correctly.")
                print(f"   Parsed: {obj.model_dump(exclude_unset=True)}")
            else:
                print(f"FAILURE: Expected an error but got success.")

        except ValidationError as e:
            if should_pass:
                print(f"FAILURE: Expected success but got error:")
                print(e.json(indent=2)) # Print detailed Pydantic error
            else:
                print(f"SUCCESS: Caught expected error.")
                # Clean up error message for display
                errors = e.errors()
                print(f"   Error Type: {errors[0]['type']}")
                print(f"   Message: {errors[0]['msg']}")
        print("\n")

    # 1. PASS: Valid Entry
    valid_data = {
        "id": "test_001",
        "question": "What is the capital of France?",
        "options": {"A": "Berlin", "B": "Paris", "C": "Madrid"},
        "answer_idx": "B",
        "source": "Geography101"
    }
    run_test("Valid Entry", valid_data, should_pass=True)

    # 2. FAIL: Empty Mandatory String (Question is whitespace)
    empty_string_data = {
        "id": "test_002",
        "question": "   ", 
        "options": {"A": "1", "B": "2"},
        "answer_idx": "A"
    }
    run_test("Empty String Check", empty_string_data, should_pass=False)

    # 3. FAIL: Answer Key Logic (Key 'D' missing from options)
    bad_logic_data = {
        "id": "test_003",
        "question": "Valid Question",
        "options": {"A": "1", "B": "2"},
        "answer_idx": "D"
    }
    run_test("Logic Check (Key Missing)", bad_logic_data, should_pass=False)

    # 4. FAIL: Not Enough Options
    short_options_data = {
        "id": "test_004",
        "question": "Valid Question",
        "options": {"A": "1"}, # Only 1 option
        "answer_idx": "A"
    }
    run_test("Minimum Options Check", short_options_data, should_pass=False)

    # 5. FAIL: Type Error (Options is a list, not a dict)
    bad_type_data = {
        "id": "test_005",
        "question": "Valid Question",
        "options": ["A", "B"], # List instead of Dict
        "answer_idx": "A"
    }
    run_test("Type Check", bad_type_data, should_pass=False)

    # 6. FAIL: Missing Mandatory Field
    missing_field_data = {
        "id": "gen_006",
        "question": "Valid Question"
        # options and answer_idx are missing
    }
    run_test("Missing Fields", missing_field_data, should_pass=False)
