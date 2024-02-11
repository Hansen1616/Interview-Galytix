import traceback
import sys

from helper_galytix_interview import (
    InterviewHelper,
    FileExistsException,
)

try:
    helper = InterviewHelper(run_testing=True)
    helper.init_files()
    helper.print_comparison_of_phrases()
    closest_phrase = helper.print_comparison_of_user_custom_phrase("How does this phrase compare to other phrases?")


except FileExistsException as e:
    trace = traceback.format_exc()
    err_msg = str(e) + "\n Full Error Message: \n" + trace
    helper.log_progress(f"{err_msg}")
    sys.exit(1)


except Exception as e:
    trace = traceback.format_exc()
    err_msg = "An unexpected Error occured: \n" + str(e) + "\n Full Error Message: \n" + trace
    helper.log_progress(f"{err_msg}")
    sys.exit(1)
