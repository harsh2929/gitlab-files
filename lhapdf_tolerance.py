import lhapdf
import numpy as np

def test_tolerance_values():
    tolerances = [1e-5, 1e-6, 1e-7]

    x = np.linspace(1e-8, 1, 10)
    Q = np.linspace(1, 100000, 10)
    for tol in tolerances:
        pdf = lhapdf.mkPDF("NNPDF30_nlo_as_0118", 0)
        pdf.set_tolerance(tol)

        for xi in x:
            for Qi in Q:
                expected_val = pdf.xfxQ(1, xi, Qi)
                actual_val = pdf.xfxQ(1, xi, Qi, error=False)
                assert np.isclose(expected_val, actual_val, rtol=1e-6, atol=1e-6)
    for tol in tolerances:
        pdf = lhapdf.mkPDF("NNPDF30_nlo_as_0118", 0)
        pdf.set_tolerance(tol)

        assert "Tolerance: {}".format(tol) in pdf.description()
