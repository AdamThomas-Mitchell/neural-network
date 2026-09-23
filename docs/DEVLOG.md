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
