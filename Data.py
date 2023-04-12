import csv

def write_data(output, length, bitrate, horizontal_aspect, vertical_aspect):

    with open('dataset/dataset.csv', 'a+', newline='') as f:

        writer = csv.writer(f)

        if f.tell() == 0:

            writer.writerow(["output size in bytes", "length in seconds", "bitrate", "horizontal aspect", "vertical aspect"])

        writer.writerow([output, length, bitrate, horizontal_aspect, vertical_aspect])

