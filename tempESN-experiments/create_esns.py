import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs
import NymphESN.restrictedmatrix as rmatrix
import tempESN.TempESN as temp

subreservoir_N = 64
DW = 0.7
DB = 0.1
i = 1

# MSO 2
rhythm_two = np.array([[1],[0, 1]])

W = rmatrix.create_restricted_esn_weights(subreservoir_N*2, subreservoir_N, 2, DW, DB)
standard = nymph.NymphESN(1, subreservoir_N*2, 1, density=DW, seed=i)
restricted = nymph.NymphESN(1, subreservoir_N*2, 1, density=DW, seed=i)
restricted.set_weights(W)
gondor_encodings = temp.TempESN_Encoding.generate_gondor_encoding()
circuit_encodings = temp.TempESN_Encoding.generate_circuit_encoding()
gondor = temp.Temporal_ESN(1, subreservoir_N*2, 1, gondor_encodings, seed=i)
gondor.set_rhythms(rhythm_two)
gondor.set_weights(W)
circuit = temp.Temporal_ESN(1, subreservoir_N*2, 1, circuit_encodings, seed=i)
circuit.set_rhythms(rhythm_two)
circuit.set_weights(W)




rhythm_three = np.array([[1], [0, 1], [0, 0, 1]])

W = rmatrix.create_restricted_esn_weights(subreservoir_N*2, subreservoir_N, 2, DW, DB)
standard = nymph.NymphESN(1, subreservoir_N*2, 1, density=DW, seed=i)
restricted = nymph.NymphESN(1, subreservoir_N*2, 1, density=DW, seed=i)
restricted.set_weights(W)
gondor_encodings = temp.TempESN_Encoding.generate_gondor_encoding()
circuit_encodings = temp.TempESN_Encoding.generate_circuit_encoding()
gondor = temp.Temporal_ESN(1, subreservoir_N*2, 1, gondor_encodings, seed=i)
gondor.set_rhythms(rhythm_three)
gondor.set_weights(W)
circuit = temp.Temporal_ESN(1, subreservoir_N*2, 1, circuit_encodings, seed=i)
circuit.set_rhythms(rhythm_three)
circuit.set_weights(W)


rhythm_four = np.array([[1], [0, 1], [0, 0, 1], [0, 0, 0, 1]])

W = rmatrix.create_restricted_esn_weights(subreservoir_N*2, subreservoir_N, 2, DW, DB)
standard = nymph.NymphESN(1, subreservoir_N*3, 1, density=DW, seed=i)
restricted = nymph.NymphESN(1, subreservoir_N*3, 1, density=DW, seed=i)
restricted.set_weights(W)
gondor_encodings = temp.TempESN_Encoding.generate_gondor_encoding()
circuit_encodings = temp.TempESN_Encoding.generate_circuit_encoding()
gondor = temp.Temporal_ESN(1, subreservoir_N*3, 1, gondor_encodings, seed=i)
gondor.set_rhythms(rhythm_four)
gondor.set_weights(W)
circuit = temp.Temporal_ESN(1, subreservoir_N*3, 1, circuit_encodings, seed=i)
circuit.set_rhythms(rhythm_four)
circuit.set_weights(W)

rhythm_eight = np.array([[1], [0, 1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1]])

W = rmatrix.create_restricted_esn_weights(subreservoir_N*4, subreservoir_N, 2, DW, DB)
standard = nymph.NymphESN(1, subreservoir_N*4, 1, density=DW, seed=i)
restricted = nymph.NymphESN(1, subreservoir_N*4, 1, density=DW, seed=i)
restricted.set_weights(W)
gondor_encodings = temp.TempESN_Encoding.generate_gondor_encoding()
circuit_encodings = temp.TempESN_Encoding.generate_circuit_encoding()
gondor = temp.Temporal_ESN(1, subreservoir_N*4, 1, gondor_encodings, seed=i)
gondor.set_rhythms(rhythm_eight)
gondor.set_weights(W)
circuit = temp.Temporal_ESN(1, subreservoir_N*4, 1, circuit_encodings, seed=i)
circuit.set_rhythms(rhythm_eight)
circuit.set_weights(W)
