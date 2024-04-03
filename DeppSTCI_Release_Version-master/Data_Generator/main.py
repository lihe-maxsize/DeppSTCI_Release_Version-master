from data_generator import Data_generator
import multiprocessing

def data_generate_API(T, M, N, C, base):
    generator = Data_generator(T=T, N=N, C=C, M=M, base=base)

    print(f"Producing_raw_data_o with para: T{T}, M{M}, N{N}, C{C}")
    generator.generating_observation()
    print(f"Success!!!! Producing_raw_data_o with para: T{T}, M{M}, N{N}, C{C}")
    print("\n********************************************\n")
    print(f"Producing_raw_data_i_with_lambda with para: T{T}, M{M}, N{N}, C{C}")
    generator.generating_intervention_with_lambda()
    print(f"Success!!!! Producing_raw_data_i_with_lambda with para: T{T}, M{M}, N{N}, C{C}")


def main():
    cfM = [1]
    cfN = [1]
    cfC = [3]
    with multiprocessing.Pool() as pool:
        pool.starmap(data_generate_API, [(64, M, N, C, "./raw_data/")
                                         for M in cfM
                                         for N in cfN
                                         for C in cfC])
    with multiprocessing.Pool() as pool:
        pool.starmap(data_generate_API, [(48, M, N, C, "./raw_data/")
                                         for M in cfM
                                         for N in cfN
                                         for C in cfC])
    with multiprocessing.Pool() as pool:
        pool.starmap(data_generate_API, [(32, M, N, C, "./raw_data/")
                                         for M in cfM
                                         for N in cfN
                                         for C in cfC])
    Train_data_gene = Data_generator(T=64, N=cfN[0], C=cfC[0], M=cfM[0], base="./raw_data/")
    Train_data_gene.dspp_data()
    Train_data_gene.ej_data()

if __name__ == '__main__':
    main()
