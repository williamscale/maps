library(reticulate)

# source_python('D:/maps/Running/pickle_reader.py')
# gpx.data <- read_pickle_file('D:/maps/Running/Data/kg_06142024_gpx.pkl')

pd <- import('pandas')
gpx.data <- pd$read_pickle('D:/maps/Running/Data/kg_06142024_gpx.pkl')
fit.data <- pd$read_pickle('D:/maps/Running/Data/kg_06142024_fit.pkl')
