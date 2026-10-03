# Development Log

## 20/09/26
Finalised the data download functionality. Probably spent too much time over optimising, but it was software dev practice. Following functionality implemented:
- can download MNIST and FASHION-MNIST datasets
- Can try multiple mirrors when downloading files for datasets
- Can verify downloaded file is correct size
- Can verify integrity of downloaded files by comparing to SHA256 hash
- Can retry downloading a given file multiple times before raising an error

Potential future improvements:
- Could implement chunking downloads for large files, although this is absolutely overkill for current datasets

Reached 100% test coverage, although this was done entirely by Claude. I have to admit, I've been slacking on the tests as I really really really hate unit tests. Need to make more of an effort though going forward to do this as I'm going rather than all at once at the end. On the Claude front - I am very impressed so far but I will make a point of not using it for any more of this project. Want to make sure I don't offload my learning to AI.

Happy to bump this up to v0.2.0 now with this functionality. Next step is data pre-processing. Need to tranform the raw data to something usable. Can probably include data visualisation in this step as well.


## 23/09/26
Working on the data preprocessing step. Decided on separate steps to first unzip the raw data file then unpack the ubyte data and save as numpy arrays. Completed the gzip step today, got Claude to write the unit tests. Next step is parsing the custom data format, this resource will be useful: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html

Once I get the individual steps done, worth coding up another cli function. Might have to update the dataset config to include transform function or something, not sure yet.


## 29/09/26
Working on handling the file handling for the uncompressed MNIST data files. Bit more complicated than anticipated with trying to read and parse the binary data from scratch. Got a reasonable idea of how to structure the code now. Got the skeleton down, next step is to implement the binary data reader class.


## 01/10/26
Firs implementation of the binary reader done. Making a bit more sense now. Next step is to write some tests, could probably optimise a little bit as well.

Turned off the MyPy pre-commit stuff as it's annoying me. Need to come back and sort this out.


## 02/10/26
Made sure binary reader can read the necessary data types that are stated in the IDX format specification. Had a think about how to model the IDX files, have a first pass at the idx models module. Implemented a funcction in the parser module to read the header. This has me thinking about whether my model for the data record is quite right. Instead of one global array it's probably better to model the data as a list or generator of individual records. Need to make a first pass in the parser module for reading the actual data in the IDX files, should play about with this in a Jupyter notebook. Might need to add a skip_bytes function to the binary reader. Also would be useful to know how many bytes the header is, but feels wrong to include that in the header class when it's not explicitly stated in the idx file. Will have a think. It'll become clearer as a crack on through the parser module I'm sure.

Next step: first pass at loading data in the parser module.


## 03/10/26
Ok think I've largely sorted the main functionality for data processing. Could do with a tidy up, and I need to write it into a CLI script. Refactored things to improve the feng shui.

Couple things to note:
- Binary reader, does not cache struct types. This could offer a speed up but actually we only need the binary reader for reading the header of the IDX files so would probably be overkill anyway. Still would be informative to read up on lru_cache stuff.
- Currently exists a function to parse an individual sample from an IDX file. Can't image this will ever be used as I settled on just loading all data. Too sentimental to remove.
- On that note, when converting IDX file to numpy I'm just loading everything into memory. Could probably make that more robust by iterating through samples and saving to np file but the dataset is pretty small and this feels like it would be an over-optimisation.
- Currently in the processing step I'm unzipping gzip then writing to IDX then opening IDX and writing a NumPy file. This could probably be streamlined. I wonder if i could even just go straight to compressed file to np array, seems numpy has some functions that could be useful for that. Fine for now though if it works.

Next steps:
1. Tidy up current processing code
2. Write CLI script for data processing
3. Tests (sad)
4. Data visualisation utility function for image samples
5. Error handling for existing code



### Actions:
- Properly configure MyPy pre-commit hook.
- Set up Claude code on GitHub to review PRs.
- Tidy up GitHub project.
