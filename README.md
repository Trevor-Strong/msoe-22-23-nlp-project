## Install

Run the following shell commands to setup the virtual environment and install dependencies

```powershell
python -m venv venv             # initializes the virtual environment
./venv/scripts/activate         # Activates the virtual environment
pip install -r requirements.txt # installs dependencies
```

## Running

The run script is `src/main.py`, you give it some text, in the terminal, as input and it
returns to you the parts of speech it thinks each word is. You can use `CTL+C` to exit the
program at any time.

### Example
Input: `A sentence about something.`

Output: `[['DET', 'NOUN', 'ADP', 'NOUN', '.']]`

This means that our model thinks:
- `A` is a determinate
- `sentence` is a noun
- `about` is an adposition
- `something` is a noun
- `.` is punctuation

### Tags to parts of speach
 - `VERB` - verbs (all tenses and modes)
 - `NOUN` - nouns (common and proper)
 - `PRON` - pronouns
 - `ADJ` - adjectives
 - `ADV` - adverbs
 - `ADP` - adpositions (prepositions and postpositions)
 - `CONJ` - conjunctions
 - `DET` - determiners
 - `NUM` - cardinal numbers
 - `PRT` - particles or other function words
 - `X` - other: foreign words, typos, abbreviations
 - `.` - punctuation
