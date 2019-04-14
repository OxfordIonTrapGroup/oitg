from .gate import GateSequence
from .qasm import gate_experiment_to_qasm, stringify_qasm


def save_circuit_pdf(filename: str, gates: GateSequence):
    """Render a gate sequence as a circuit diagram and save it to PDF.

    This requires Matplotlib and Qiskit Terra to be installed.

    :param filename: Path to the output PDF file.
    :param gates: The gate sequence to display.
    """
    # TODO: This should be made more configurable, but at the time of writing, the
    # Qiskit drawing code only seems to work when writing to PDF.
    import qiskit
    qasm = stringify_qasm(gate_experiment_to_qasm(gates))
    qcircuit = qiskit.converters.dag_to_circuit(
        qiskit.converters.ast_to_dag(qiskit.qasm.Qasm(data=qasm).parse()))
    qcircuit.draw(output="mpl",
                  filename=filename,
                  style={
                      "usepiformat": True,
                      "cregbundle": True
                  })
