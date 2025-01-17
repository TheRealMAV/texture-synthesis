from . import texture_synthesis
from . import object_removal


def run_all():
    texture_synthesis.main()
    object_removal.main()
    print("The results are ready!")


if __name__ == "__main__":
    run_all()
