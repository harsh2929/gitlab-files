import pytest
import lhapdf 
import numpy as np

@pytest.fixture(scope="module")
def pdf():
    return lhapdf.getPDFSet("NNPDF40_nnlo_as_0118").mkPDF(0)
def test_pdf_at_one(pdf):
    q_values = [10, 100, 1000]
    for q in q_values:
        for pid in range(-6, 7):
            value = pdf.xfxQ(pid, 1, q)
            assert value == 0



def test_pdf_pid_range(pdf):
    x_values = [0.1, 0.5, 0.9]
    q_values = [10, 100, 1000]
    for x in x_values:
        for q in q_values:
            for pid in [-7, -100, 7, 100]:
                with pytest.raises(RuntimeError):
                    pdf.xfxQ(pid, x, q)
def test_pdf_values(pdf):
    assert np.isclose(pdf.xfxQ(0, 1.0, 10.0), 0.0, rtol=1e-9)

def test_pdf_stability(pdf):
    pdf1 = pdf
    pdf2 = lhapdf.getPDFSet("NNPDF40_nnlo_as_0118").mkPDF(0)
    max_diff = np.max(np.abs((pdf1.xf - pdf2.xf) / pdf1.xf))

    assert max_diff < 1e-9
def test_pdf_values_for_all_partons(pdf):
    for i in range(-6, 7):
        value = pdf.xfxQ(i, 0.5, 10.0)
        assert np.isfinite(value)
def test_pdf_values_at_random_points(pdf):
    np.random.seed(123)
    for i in range(10):
        x = np.random.uniform(1e-8, 1)
        Q = np.random.uniform(1, 1e5)
        value = pdf.xfxQ(0, x, Q)
        assert np.isfinite(value)            