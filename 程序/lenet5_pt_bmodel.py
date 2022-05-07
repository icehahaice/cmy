import bmnetp


def main():
    bmnetp.compile(
        model="./lenet.zip",
        shapes=[[1, 1, 32, 32]],
        net_name="lenet5",
        outdir="./LeNet5_PT_bmodel",
        target="BM1684",
        opt=2
    )


if __name__ == '__main__':
    main()
