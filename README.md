<p align="center">
  <img width="640" height="480" src="./results/general/tform_progression.gif">
</p>

<h1 align="center">Pythostitcher</h2>

<p align="center">
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/badge/linting-pylint-yellowgreen></a>
</p>
    
## General
Pythostitcher is a Python implementation of the AutoStitcher software which stitches prostate histopathology images into a pseudo whole mount image. For more information about AutoStitcher, check out the original paper [here](https://www.nature.com/articles/srep29906) or find the original Matlab implementation [here](https://engineering.case.edu/centers/ccipd/content/software).

## Algorithm
The input for Pythostitcher consists of four quadrant images which are labeled according to their original location (upper left, upper right, lower left, lower right). Pythostitcher will then use this location to automatically rotate the quadrants and compute a very coarse initial alignment. This initial alignment is then iteratively refined with increasingly finer resolutions using a genetic algorithm. One of the strengths of Pythostitcher is that the optimal alignment between quadrants can be scaled linearly for finer resolutions. Hence, when a satisfactory alignment is achieved on a lower resolution, this alignment can be scaled up linearly to compute the full resolution stitched result. 

## User instructions
You can try out Pythostitcher yourself by running the main script on the provided sample data. Read the docstring in the main.py file for further instructions. The gif above is the approximate result you should get for P000001 from the sample data. Note that your result may differ slightly due to the randomly generated mutations in every generation of the genetic algorithm. 

## Licensing
The source code of Pythostitcher is licensed under the [GNU Lesser General Public License (LGPL)](https://www.gnu.org/licenses/lgpl-3.0.nl.html). The provided sample data is licensed under the [CC-BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/3.0/). Please take these licenses into account when using Pythostitcher.
