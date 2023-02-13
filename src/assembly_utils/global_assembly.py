from .global_assembly_utils import *


def global_assembly(parameters):
    """
    Main function to call the assembly script.
    """

    parameters["log"].log(parameters["my_level"], f" - exploring feasible assemblies")

    # Perform main assembly
    print("\nPerforming global assembly...")

    # Load class instance
    case = Assembler(parameters)

    # Verify that all required files are present
    case.check_case_eligibility()

    # Process input files
    case.process_input_files()
    case.process_input_images()

    # Obtain all feasible configurations
    case.get_feasible_configurations()

    # Obtain all solutions for each possible configuration
    case.get_solutions_per_configuration()

    # Evaluate all solutions for plausibility
    case.evaluate_solutions()
    parameters["log"].log(parameters["my_level"], f" - finished!\n")

    return
