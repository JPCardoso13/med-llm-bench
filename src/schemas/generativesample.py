from typing import Optional, Dict
from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from dataclasses import field

class GenerativeSample(BaseModel):
    """
    A unified representation of an Open-Ended (Generative) Question.

    Used for tasks where the model must generate a free-text response
    without provided options.

    Attributes:
        id (str): A unique identifier for the sample.
        question (str): The question the model should answer..
        ref_answer (str): The reference answer (ground truth) used for evaluation.
        source (Optional[str]): The origin dataset name (e.g., "medqa", "mmlu", "custom_test").
                                Defaults to None.
        category (Optional[str]): The specific domain, topic, or sub-field. 
                                  Example: "Cardiology", "Physics". Defaults to None.
        tags (Dict[str, str]): A dictionary of arbitrary key-value pairs for additional metadata.
                               Defaults to an empty dictionary.
        context (Optional[str]): Reference text if the task is context-dependent
                                 (e.g., a reading comprehension task). Defaults to None.
    
    Raises:
        ValidationError: If any of the constraints are violated, a detailed error message will indicate the issue.
    """
    id: str
    question: str
    ref_answer: str
    source: Optional[str] = None
    category: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    context: Optional[str] = None

    # Constraint 1: Mandatory fields must not be empty or whitespace
    @field_validator('id', 'question', 'ref_answer')
    @classmethod
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return v

if __name__ == "__main__":
    def run_test(name: str, data: dict, should_pass: bool):
        print(f"--- Test: {name} ---")
        try:
            obj = GenerativeSample(**data)

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
        "id": "gen_001",
        "question": "Explain quantum entanglement.",
        "ref_answer": "Quantum entanglement is a phenomenon where...",
        "source": "PhysicsDB",
        "category": "Science"
    }
    run_test("Valid Entry", valid_data, should_pass=True)

    # 2. FAIL: Empty Question
    empty_q_data = {
        "id": "gen_002",
        "question": "", 
        "ref_answer": "Valid answer"
    }
    run_test("Empty Question", empty_q_data, should_pass=False)

    # 3. FAIL: Whitespace Reference Answer
    whitespace_data = {
        "id": "gen_003",
        "question": "Valid Question",
        "ref_answer": "   "  # Just whitespace
    }
    run_test("Whitespace Ref Answer", whitespace_data, should_pass=False)

    # 4. FAIL: Missing Mandatory Field
    missing_field_data = {
        "id": "gen_004",
        "question": "Valid Question"
        # ref_answer is missing
    }
    run_test("Missing Field", missing_field_data, should_pass=False)

    # 5. PASS: Valid Tags
    valid_tags_data = {
        "id": "gen_005",
        "question": "Explain quantum entanglement.",
        "ref_answer": "Quantum entanglement is a phenomenon where...",
        "tags": {"specialty": "physics", "difficulty": "medium"}
    }
    run_test("Valid Tags", valid_tags_data, should_pass=True)

    # 6. FAIL: Tags is List (not Dict)
    tags_list_data = {
        "id": "gen_006",
        "question": "Valid Question",
        "ref_answer": "Valid answer",
        "tags": ["tag1", "tag2"]  # List instead of Dict
    }
    run_test("Tags Type Check (List)", tags_list_data, should_pass=False)

    # 7. FAIL: Tags Dict with Non-String Values
    tags_bad_value_data = {
        "id": "gen_007",
        "question": "Valid Question",
        "ref_answer": "Valid answer",
        "tags": {"difficulty": 5}  # Int value instead of String
    }
    run_test("Tags Value Type Check", tags_bad_value_data, should_pass=False)
