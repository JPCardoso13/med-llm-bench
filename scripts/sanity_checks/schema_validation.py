from typing import Any, Dict, Type
from pydantic import BaseModel, ValidationError

from llm_bench.schemas import GenerativeSample, MCQSample


def run_validation_case(
    model_cls: Type[BaseModel],
    name: str,
    data: Dict[str, Any],
    should_pass: bool,
) -> None:
    print(f"--- {model_cls.__name__}: {name} ---")
    try:
        obj = model_cls(**data)
        if should_pass:
            print("SUCCESS: Data validated correctly.")
            print(f"   Parsed: {obj.model_dump(exclude_unset=True)}")
        else:
            print("FAILURE: Expected an error but got success.")
    except ValidationError as exc:
        if should_pass:
            print("FAILURE: Expected success but got error:")
            print(exc.json(indent=2))
        else:
            print("SUCCESS: Caught expected error.")
            errors = exc.errors()
            print(f"   Error Type: {errors[0]['type']}")
            print(f"   Message: {errors[0]['msg']}")
    print()


def run_generative_cases() -> None:
    run_validation_case(
        GenerativeSample,
        "Valid Entry",
        {
            "id": "gen_001",
            "question": "What is the first-line treatment for community-acquired pneumonia in a stable outpatient adult?",
            "answer": "Empiric oral antibiotics such as amoxicillin or doxycycline.",
            "grouping": {
                "specialty": ["infectious_disease", "pulmonology"],
                "medical_task": ["treatment_selection"],
            },
            "metadata": {
                "article_link": "https://example.org/cap-guideline",
                "pmc_id": "PMC123456",
            },
        },
        should_pass=True,
    )

    run_validation_case(
        GenerativeSample,
        "Grouping Must Be List Values",
        {
            "id": "gen_002",
            "question": "Valid question",
            "answer": "Valid answer",
            "grouping": {"specialty": "cardiology"},
        },
        should_pass=False,
    )

    run_validation_case(
        GenerativeSample,
        "Empty Question",
        {
            "id": "gen_003",
            "question": "   ",
            "answer": "Valid answer",
        },
        should_pass=False,
    )

    run_validation_case(
        GenerativeSample,
        "Invalid Grouping Type",
        {
            "id": "gen_004",
            "question": "Which imaging modality is preferred first for suspected acute appendicitis in children?",
            "answer": "Ultrasound is typically the preferred first-line imaging modality.",
            "grouping": ["specialty"],
        },
        should_pass=False,
    )


def run_mcq_cases() -> None:
    run_validation_case(
        MCQSample,
        "Valid Entry",
        {
            "id": "mcq_001",
            "question": "Which marker is most specific for myocardial injury?",
            "options": {
                "A": "Troponin I",
                "B": "D-dimer",
                "C": "C-reactive protein",
                "D": "BNP",
            },
            "answer_idx": "A",
            "grouping": {
                "specialty": ["cardiology"],
                "medical_task": ["diagnostic_interpretation"],
            },
            "metadata": {
                "source_url": "https://example.org/acs-reference",
                "pmc_id": "PMC654321",
            },
        },
        should_pass=True,
    )

    run_validation_case(
        MCQSample,
        "Options Dict Has Fewer Than Two Entries",
        {
            "id": "mcq_001b",
            "question": "Which finding is most concerning for sepsis?",
            "options": {"A": "Fever"},
            "answer_idx": "A",
            "grouping": {
                "specialty": ["critical_care"],
                "medical_task": ["risk_stratification"],
            },
        },
        should_pass=False,
    )

    run_validation_case(
        MCQSample,
        "Grouping Must Be List Values",
        {
            "id": "mcq_002",
            "question": "Which drug class is first-line for chronic heart failure with reduced ejection fraction?",
            "options": {"A": "ACE inhibitor", "B": "H1 antihistamine"},
            "answer_idx": "A",
            "grouping": {"specialty": "cardiology"},
            "metadata": {"difficulty": "medium"},
        },
        should_pass=False,
    )

    run_validation_case(
        MCQSample,
        "Answer Missing From Options",
        {
            "id": "mcq_003",
            "question": "What is the recommended immediate treatment for anaphylaxis?",
            "options": {"A": "IV fluids", "B": "Antihistamine only"},
            "answer_idx": "D",
        },
        should_pass=False,
    )

    run_validation_case(
        MCQSample,
        "Grouping Has Non-String List Values",
        {
            "id": "mcq_004",
            "question": "Which lab abnormality is classically associated with primary hyperparathyroidism?",
            "options": {"A": "Hypercalcemia", "B": "Hypocalcemia"},
            "answer_idx": "A",
            "grouping": {"specialty": [1]},
        },
        should_pass=False,
    )


if __name__ == "__main__":
    run_generative_cases()
    run_mcq_cases()
