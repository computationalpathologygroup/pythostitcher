<p align="center">
  <img width="640" height="480" src="./tform_progression.gif">
</p>

<h1 align="center">Pythostitcher</h2>

<p align="center">
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/badge/linting-pylint-yellowgreen></a>
</p>
    
## General
Pythostitcher is a Python implementation of the [AutoStitcher](https://www.nature.com/articles/srep29906) software which stitches prostate histopathology images into an artifical whole mount image. Although the general principle of Pythostitcher is very similar to AutoStitcher, PythoStitcher is implemented in Python and significantly faster compared to the [original implementation in Matlab](https://engineering.case.edu/centers/ccipd/content/software). In addition, Pythostitcher offers several new advanced features such as 1) saving the end result at maximum resolution (0.25 µm pixel spacing), 2) providing a smooth overlap between overlapping parts due to gradient alpha blending and 3) providing support for a varying number of tissue fragments. 

## Algorithm
The input for Pythostitcher consists of either two or four images which are labeled according to their desired location and rotation (in steps of 90 degrees). In the case of a prostatectomy cross-section which has been sliced in four fragments, these locations would be upper left, upper right, lower left, lower right. Pythostitcher will then use this location and rotation to perform an automated edge detection and compute a very coarse initial alignment. This initial alignment is then iteratively refined with increasingly finer resolutions using a genetic algorithm. One of the strengths of Pythostitcher is that the optimal alignment between fragments can be scaled linearly for finer resolutions. Hence, when a satisfactory alignment is achieved on a lower resolution, this alignment can be scaled up linearly to compute the full resolution stitched result. 

## User instructions
You can try out Pythostitcher yourself on the sample data available from <a href="https://doi.org/10.5281/zenodo.7002505"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7002505.svg" alt="DOI"></a>. This sample data includes two prostatectomy cases with four fragments and one oesophagectomy case with two fragments. After downloading the data, place each case in the respective 'tissue directory' and ensure that the data is now structured as follows:
	
	pythostitcher/ 
	├── src
	├── sample_data
    │     └── prostate
	│        ├── P000001
	│        └── P000002
    │     └── oesophagus
    │         └── P000001

            
You can run PythoStitcher through the command line using:

    python3 main.py --tissue {tissue_type} --patient {patient_idx} 
where *tissue_type* and *patient_idx* are used to find the data for a specific patient in the sample_data directory. Note that *tissue_type* should match the name of any of the directories in the sample_data directory and the *patient_idx* should be represented as an integer. Example line to obtain the result for patient P000001 from the prostate dataset:

    python3 main.py --tissue "prostate" --patient 1

 

## Licensing
The source code of Pythostitcher is licensed under the [GNU Lesser General Public License (LGPL)](https://www.gnu.org/licenses/lgpl-3.0.nl.html). The provided sample data is licensed under the [CC Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode). Please take these licenses into account when using Pythostitcher.

