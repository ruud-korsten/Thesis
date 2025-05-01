from quality_assessment.note_engine import NoteEngine
from quality_assessment.note_evaluator import NoteEvaluator
from quality_assessment.note_inspector import inspect_note_violations

def handle_note_fallbacks(domain_notes: list[str], fallback_notes: list[str], df) -> dict:
    all_notes = domain_notes + fallback_notes
    if not all_notes:
        print("No domain or fallback notes to process.")
        return {}

    print("\nRunning Note Engine on domain and fallback notes...")
    engine = NoteEngine(model="gpt-4o")
    note_functions = engine.run(all_notes, df.columns.tolist())

    print("\nEvaluating generated note checks...")
    evaluator = NoteEvaluator(df)
    note_results = evaluator.evaluate(note_functions)

    for note, result in note_results.items():
        print("\n---")
        print(f"Note: {note}")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"✓ Function: {result['function_name']}")
            print(f"→ Violations flagged: {result['violations']}")

    return note_results
