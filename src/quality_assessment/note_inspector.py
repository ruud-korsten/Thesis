import pandas as pd

def inspect_note_violations(df: pd.DataFrame, results: dict):
    while True:
        print("\nAvailable note checks:")
        valid_notes = [note for note in results if "violations" in results[note]]
        for i, note in enumerate(valid_notes):
            print(f"  {i+1}. {note} ({results[note]['violations']} violations)")

        selection = input("\nEnter a number to inspect (or press Enter to exit): ").strip()
        if not selection:
            print("Exiting note inspection.")
            break

        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(valid_notes):
                print("Invalid selection.")
                continue

            note = valid_notes[idx]
            func_name = results[note]['function_name']
            code = results[note]['code']

            print(f"\nFunction: {func_name}")
            print(f"Code:\n{code}")

            # Re-execute function to get mask
            local_env = {}
            exec(code, {}, local_env)
            func = next(v for v in local_env.values() if callable(v))
            mask = func(df)

            print("\nFirst 5 violations:")
            print(df[mask].head())

        except Exception as e:
            print(f"Error during inspection: {e}")
