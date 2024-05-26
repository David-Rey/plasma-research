"""
This library is for SPE-Files created by LightField 5.3 from Princeton Instruments
It is designed to work with SPE2.x and SPE3.0-Files and requires Python 3.

https://github.com/HydraSpex/SPE_Handler
"""


import struct
import numpy as np
import math
import os


#Reading Bytes
def from_bytes(b, format, offset):
    calcsize = struct.calcsize(format)
    return struct.unpack(format, b[offset:offset + calcsize])[0]


#Reading the XML Line of SPE3.x Files
def openXMLline(filename):
    num_lines = sum(1 for line in open(filename, "rb"))
    f = open(filename, "rb")
    lines = f.readlines()

    Txt_Dai = open("Dai_new.txt", "w")

    f = open(filename, "rb")
    lines = f.readlines()

    i = 0
    while i < num_lines:
        Txt_Dai.write(str(lines[i]) + "\n")
        i += 1
    Txt_Dai.close()

    XMLline = str(lines[num_lines - 1])
    startXML = XMLline.find("<SpeFormat")
    XMLline = XMLline[startXML:]

    PosVersion = XMLline.find("version") + 9
    Version = XMLline[PosVersion:PosVersion + 3]

    PosFrame = XMLline.find("Frame") + 14
    Frame = XMLline[PosFrame:]
    PosFrameEnd = Frame.find("\"")
    Frame = Frame[:PosFrameEnd]

    PosWidth = XMLline.find("width") + 7
    Width = XMLline[PosWidth:]
    PosWidthEnd = Width.find("\"")
    Width = Width[:PosWidthEnd]

    PosHeight = XMLline.find("height") + 8
    Height = XMLline[PosHeight:]
    PosHeightEnd = Height.find("\"")
    Height = Height[:PosHeightEnd]

    PosLaser = XMLline.find("WavelengthLaserLine")
    Laser = XMLline[PosLaser:]
    PosLaser = Laser.find("\">") + 2
    Laser = Laser[PosLaser:]
    PosLaserEnd = Laser.find("<")
    Laser = Laser[:PosLaserEnd]

    PosCreate = XMLline.find("created") + 9
    Created = XMLline[PosCreate:]
    PosTime = Created.find("T")
    Date = Created[:PosTime]
    PosTimeEnd = Created.find(".")
    Time = Created[PosTime + 1:PosTimeEnd]

    PosWave = XMLline.find("<Wavelength ")
    PosWaveEnd = XMLline.find("</Wavelength>")
    WaveData = XMLline[PosWave:PosWaveEnd]
    WaveData = WaveData[(WaveData.find("\">") + 2):]

    Wavedata = []
    WavedataRound = []
    i = 0

    while i < int(Width):
        PosNext = WaveData.find(",")
        Wavedata.append(WaveData[:PosNext])
        WavedataRound.append(round(float(WaveData[:PosNext]), 2))
        WaveData = WaveData[PosNext + 1:]
        i += 1

    PosCWL = XMLline.find("<CenterWavelength")
    PosCWLEnd = XMLline.find("</CenterWavelength>")
    CWLData = XMLline[PosCWL:PosCWLEnd]
    CWL = CWLData[(CWLData.find("\">") + 2):]

    PosBG = (XMLline.find("<BackgroundCorrection><Enabled") + 25)
    BGData = XMLline[PosBG:]
    PosBGBegin = (BGData.find(">") + 1)
    PosBGEnd = BGData.find("<")
    BG = BGData[PosBGBegin:PosBGEnd]

    PosGrating = XMLline.find("<Grating><Selected")
    GratingData = XMLline[PosGrating:]
    PosGratingBegin = GratingData.find("[")
    PosGratingEnd = (GratingData.find("]") + 1)
    Grating = GratingData[PosGratingBegin:PosGratingEnd]
    '''ExpTime,'''
    return Version, Frame, Width, Height, Laser, Date, Time, CWL, Grating, BG, Wavedata, WavedataRound


#Converting the SPE-Data to TXT-File.
#If space!="tab" the spacer between the values will be ";".
#If header=False the TXT-File will not contain the important setup information.
#If invert=True the TXT-File will be inverted.
def convert_txt(filename, FolderName, spe_file_name, space="tab", header=True, invert=False):
    print("Converting...")
    File = FolderName + f"/{spe_file_name}.txt"
    Txt_Point = open(File, "w")

    np_type, itemsize, Count, Version, Frame, Width, Height, Laser, Date, Time, CWL, Grating, BG, Wavedata, WavedataRound = getData(
        filename)

    if header == True:
        Txt_Point.write("#SPE version: " + str(Version) + "\n")
        Txt_Point.write("#Frame width: " + str(Width) + "\n")
        Txt_Point.write("#Frame height: " + str(Height) + "\n")
        Txt_Point.write("#Number of frames: " + str(Frame) + "\n")
        #Txt_Point.write("#Exposure (s): " + str(ExpTime) + "\n")
        Txt_Point.write("#Laser Wavelength (nm): " + str(Laser) + "\n")
        Txt_Point.write("#Central Wavelength (nm): " + str(CWL) + "\n")
        Txt_Point.write("#Date collected: " + str(Date) + "\n")
        Txt_Point.write("#Time collected (hhmmss): " + str(Time) + "\n")

    Width = int(Width)
    Height = int(Height)
    Frame = int(Frame)

    file = open(filename, "rb")
    bytes = file.read()

    datatype = from_bytes(bytes, "h", 108)
    to_np_type = [np.float32, np.int32, np.int16, np.uint16, None, np.float64, np.uint8, None, np.uint32]
    np_type = to_np_type[datatype]
    itemsize = np.dtype(np_type).itemsize

    Count = Width * Height

    Px = int(math.sqrt(Frame))
    data = []
    for i in range(0, Frame):
        data.append(np.frombuffer(bytes, dtype=np_type, count=Count, offset=4100 + i * Count * itemsize))

    if invert == False:
        Txt_Point.write("Wavelength\t")
        for j in range(0, Width):
            Txt_Point.write(Wavedata[j] + "\t")
        Txt_Point.write("\n")

        if space == "tab":
            for i in range(0, Frame):
                Txt_Point.write("Frame " + str(i + 1))
                for j in range(0, Width):
                    Txt_Point.write("\t" + str(data[i][j]))
                Txt_Point.write("\n")
        else:
            for i in range(0, Frame):
                Txt_Point.write("Frame " + str(i + 1))
                for j in range(0, Width):
                    Txt_Point.write("; " + str(data[i][j]))
                Txt_Point.write("\n")
    else:
        Txt_Point.write("Wavelength\t")
        for j in range(0, Frame):
            Txt_Point.write("Frame " + str(j + 1) + "\t")
        Txt_Point.write("\n")

        if space == "tab":
            for i in range(0, Width):
                Txt_Point.write(str(Wavedata[i]))
                for j in range(0, Frame):
                    Txt_Point.write("\t" + str(data[j][i]))
                Txt_Point.write("\n")
        else:
            for i in range(0, Width):
                Txt_Point.write(str(Wavedata[i]))
                for j in range(0, Frame):
                    Txt_Point.write("; " + str(data[j][i]))
                Txt_Point.write("\n")

    Txt_Point.close()
    file.close()


#Getting all the important setup informations out of the SPE-Header (SPE2.x), or the XML-Footer (SPE3.x)
def getData(filename):
    file = open(filename, "rb")
    bytes = file.read()

    SPEVersion = round(from_bytes(bytes, "f", 1992), 1)

    datatype = from_bytes(bytes, "h", 108)
    frame_width = from_bytes(bytes, "H", 42)
    frame_height = from_bytes(bytes, "H", 656)
    num_frames = from_bytes(bytes, "i", 1446)
    to_np_type = [np.float32, np.int32, np.int16, np.uint16, None, np.float64, np.uint8, None, np.uint32]
    np_type = to_np_type[datatype]
    itemsize = np.dtype(np_type).itemsize

    Count = frame_width * frame_height

    if SPEVersion >= 3:
        Version, Frame, Width, Height, Laser, Date, Time, CWL, Grating, BG, Wavedata, WavedataRound = openXMLline(
            filename)
    else:
        Version = SPEVersion
        Frame = num_frames
        Width = frame_width
        Height = frame_height
        Laser = from_bytes(bytes, "d", 3311)
        LocalDate = from_bytes(bytes, "16s", 20)
        LocalDate = LocalDate.decode("utf-8", "ignore")
        Day = LocalDate[:2]
        Year = LocalDate[5:9]
        Month = LocalDate[2:5]
        if Month == "Jan":
            Month = "01"
        elif Month == "Feb":
            Month = "02"
        elif Month == "Mar":
            Month = "03"
        elif Month == "Apr":
            Month = "04"
        elif Month == "May":
            Month = "05"
        elif Month == "Jun":
            Month = "06"
        elif Month == "Jul":
            Month = "07"
        elif Month == "Aug":
            Month = "08"
        elif Month == "Sep":
            Month = "09"
        elif Month == "Oct":
            Month = "10"
        elif Month == "Nov":
            Month = "11"
        elif Month == "Dec":
            Month = "12"
        else:
            Month = "00"
        Date = Year + "-" + Month + "-" + Day
        LocalTime = from_bytes(bytes, "6s", 172)
        LocalTime = LocalTime.decode("utf-8", "ignore")
        Time = LocalTime[:2] + ":" + LocalTime[2:4] + ":" + LocalTime[4:]
        UTCTime = from_bytes(bytes, "6s", 179)
        CWL = from_bytes(bytes, "f", 72)
        Grating = from_bytes(bytes, "32f", 650)
        BG = from_bytes(bytes, "i", 150)
        XStartNM = from_bytes(bytes, "d", 3183)
        XStopNM = from_bytes(bytes, "d", 3199)

        PXSize = (XStopNM - XStartNM) / (Count - 1)
        Wavedata = []
        WavedataRound = []
        j = 0
        while j < Count:
            val = XStartNM + (j * PXSize)
            Wavedata.append(val)
            WavedataRound.append(round(val, 2))
            j += 1

    return np_type, itemsize, Count, Version, Frame, Width, Height, Laser, Date, Time, CWL, Grating, BG, Wavedata, WavedataRound


def spectra_from_spe(FileName, convert=True, space="tab", header=True,
                     invert=True, referenzspektren=False):
    print("Filename:", FileName)

    #Create new Folder, if the File is converted or spectra are plotted
    PosDot = FileName.find(".spe")
    FolderName = FileName[:PosDot]
    print(FolderName)
    #find the name of the spe File
    PosSlash = FileName.rfind("/")
    SpeFileName = FileName[PosSlash + 1:PosDot]

    try:
        if referenzspektren == True:
            FolderName = f"{FileName[:PosSlash]}/txt-Files"
            os.makedirs(FolderName)
        else:
            os.makedirs(FolderName)
        print("Data folder created")
    except:
        print("Data-Folder already exist")

    if convert == True:
        convert_txt(FileName, FolderName, SpeFileName, space=space, header=header, invert=invert)
    print("done")
    return
