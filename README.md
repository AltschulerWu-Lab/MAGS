# MAGS Organoid Analysis

MAGS Organoid Analysis is the software developed for image analysis of 2d enteroid cultures [\[1\]](#ref1). This set of code accompanies a paper on 2d enteroid culture and analysis, which should be cited for this code [\[2\]](#ref2).

## Setup 

### Python

The package was developed with Python. To install the correct version of Python and required packages, follow one of the two steps.

#### Method 1
The easiest way to set up the appropriate environment is through conda. Miniconda3 can be installed at https://docs.conda.io/en/latest/miniconda.html

After installing Miniconda, download the package and navigate to the top **organoid_analysis** folder. Setup the Python environment with the following command.

```
conda env create --file=environment.yml
```

A conda environment named `mags` will be created. Activate the environment to run the scripts in this package. The environment can be activated with the following command

```
conda activate mags
```

#### Method 2
The required Python version and packages can also be installed manually. The required version and packages are listed in [environment.yml](environment.yml)

### FIJI

FIJI is used to run one of the functions in pipeline. Install FIJI at https://imagej.net/Fiji/Downloads. 

Java may be required to run FIJI in command line and can be downloaded here at http://jdk.java.net or https://adoptopenjdk.net

## References

<a name="ref1">1</a>: Thorne, C.A.\*, Chen, I.W.\*, Sanman, L.E., Cobb, M.H., Wu, L.F., and Altschuler, S.J. (2018). Enteroid Monolayers Reveal an Autonomous WNT and BMP Circuit Controlling Intestinal Epithelial Growth and Organization. Dev. Cell 44, 624–633.e4.

<a name="ref2">2</a>: Sanman, L.E.\*, Chen, I.W.\*, Bieber, J.M.\*, Steri, V., Trentesaux, C., Hann, B., Klein, O.D., Wu, L.F., and Altschuler, S.J. (2020). Transit-amplifying cells coordinate changes in intestinal epithelial cell type composition. 