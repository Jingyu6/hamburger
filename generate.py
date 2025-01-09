import lightning as L

from m2d.model.llama import M2DLlama

L.seed_everything(227)

# create model
model: M2DLlama = M2DLlama.load_from_checkpoint(
    "./local/consolidated.pt"
)

prompt = """I am a movie director and I just received the following movie plot. Could you help me answer this question? If not, let me know by writing "Not answerable". Plot title: The Dead Zone Movie plot: In the Maine small town of Castle Rock, school teacher Johnny Smith (Christopher Walken) takes his fiancée sweetheart Sarah (Brooke Adams) to an amusement park. On the roller coaster, he suffers a bad headache and tells her that he must go home to sleep it off. Sarah begs him to stay, but he declines saying: "some things are worth waiting for". As he drives home, a tanker jackknifes in front of him and he plows into it and black out.Johnny wakes up at the Weizak Clinic. His parents Herb and Vera (Sean Sullivan and Jackie Burroughs) are shown in by Dr. Sam Weizak (Herbert Lom). They tell him that he has been in a coma for five years and Sarah has married another man. The distraught Johnny is inconsolable.One day, as a nurse mops his brow he seizers her hand. He has a super-real vision of being trapped in a screaming girl's bedroom. He tells the nurse: "Amy needs your help." The nurse knows that Amy is her daughter's name, but puzzled why Johnny knows her. Johnny tells her that the house is burning, and her daughter is in the house - but it is not too late. The nurse rushes home to find her house in flames, but her daughter is rescued by firefighters just as she arrives home.After a few weeks of physical therapy, Johnny is able to walk again with some difficulty. When Johnny gratefully takes Dr Weizak's hand, he is jolted by another vision. It is wartime with burning buildings, advancing tanks. A protesting young boy is lifted into a horse-drawn cart, leaving his mother behind. Johnny tells Weizak that his mother survived. Weizak says that it is impossible, but Johnny knows her name and address. Weizak checks it out and it's true, but he cannot bring himself to speak to his elderly mother. Weizak comes back and tells Johnny: "you're either in possession of a very new human ability... or a very old one."Sarah comes to visit Johnny. Catching his stare, she asks him not to look at her that way. He says that he can't help it while Sarah says that it's been five years, but Johnny remembers it as yesterday. She says... My question: who has a car accident? The answer to this question is:"""

output = model.generate(
    prompt=prompt, 
)

print("Prompt:\n", prompt)
print("Output:\n", output)
