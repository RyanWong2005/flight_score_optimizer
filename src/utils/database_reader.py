from pathlib import Path

def load_database(file_path):

    database = {}
    units = {}

    file_path = Path(file_path)

    with open(file_path, "r") as file:

        for line_number, line in enumerate(file, start=1):

            # Remove whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Separate variable definition from unit
            definition, separator, unit = line.partition("#")

            # Extract variable name and value
            if "=" not in definition:
                raise ValueError(
                    f"Invalid entry at line {line_number}: {line}"
                )

            variable, value = definition.split("=", 1)

            variable = variable.strip()
            value = value.strip()
            unit = unit.strip() if separator else ""

            # Convert numerical values to int or float
            try:
                if "." in value or "e" in value.lower():
                    value = float(value)
                else:
                    value = int(value)

            except ValueError:
                # Keep as string if not numerical
                pass

            # Store variables and units
            database[variable] = value
            units[variable] = unit

    return database, units