import lhapdf
import numpy as np

def test_interpolation_modes():
    modes = ["linear", "spline","akima"]
    x = np.linspace(1e-8, 1, 10)
    Q = np.linspace(1, 100000, 10)
    for mode in modes:
        pdf = lhapdf.mkPDF("NNPDF30_nlo_as_0118", 0)
        pdf.interpolation = mode

        for xi in x:
            for Qi in Q:
                expected_val = pdf.xfxQ(1, xi, Qi)
                actual_val = pdf.xfxQ(1, xi, Qi, error=False)
                assert np.isclose(expected_val, actual_val, rtol=1e-6, atol=1e-6)

    # Loop over each mode and test metadata
    for mode in modes:
        pdf = lhapdf.mkPDF("NNPDF30_nlo_as_0118", 0)
        pdf.interpolation = mode
        assert pdf.interpolation == mode
        assert "Interpolation method: {}".format(mode) in pdf.description()
