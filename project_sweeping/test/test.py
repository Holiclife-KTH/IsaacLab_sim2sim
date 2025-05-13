class Person:
    def __init__(self, A=0):
        self.A=A 
        print("person")

class DK(Person):
    def __init__(self):
        super().__init__(A=)
        print("DK")
        self.A=2
if __name__ == "__main__":
    # person = Person()
    dk = DK()