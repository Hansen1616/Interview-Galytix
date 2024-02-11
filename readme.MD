# Python Excercise DE

* Resources needed for running the script can be found in [resources](resources)
* "GoogleNews-vectors-negative300.bin" needs to be extracted to the resources folder to a folder called "GoogleNews-vectors-negative300.bin"
  * i.e. full path to the file from the content root is "resources\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin"
* The full script can be found in [galytix_interview_oop](galytix_interview_oop.py)
* It uses two helper scripts
  * [helper_galytix_interview](helper_galytix_interview.py)
  * [constants](constants.py)
* Results are printed in [results](results)
  * You can find here also log files
  * It is created upon the first initialization of the helper class


## NOTES and things to improve:
- The code can be run only in the Testing mode.
- The full mode gives following error message:
    - UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 2530: character maps to <undefined>
- I was not sure how exactly to do the following requirement:
    - "Note that the whole phrase vector can be approximated by normalized sum of all the individual word tokens embeddings".
    - This could probably be also improved
- The part of normalizing the phrase vector is duplicated in two different functions
    - That could be improved to be done in one function, which would be called when needed to be used
- Currently, if a word is not found in the dictionary of words, it receives a null vector - This could be also enhanced it the future
- The function called 'init_files' could be part of the __init__ method or called at the moment when needed
  (similarly to _create_phrases_dataframe), so that it doesn't need to be called from the main function
