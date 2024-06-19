import eggshell as egg

if __name__ == "__main__":
    print(dir(egg))
    exprs = egg.io.read_expressions("data/prefix/5k_dataset.csv")
    # print(exprs)
    # print(dir(exprs[0]))
    print(exprs[0].index)
    print(exprs[0].term)

    eqsat = egg.eqsat.PyEqsatHalide(exprs[0].index)
    result = eqsat.prove_once(exprs[0].term)
    print(result.type_str())
    print(result.unpack_solved())

    # eqsat = eggshell.PyEqsatHalide(123)
    # print(eqsat)
