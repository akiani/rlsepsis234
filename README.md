# GYMIC: An OpenAI Gym Environment for Simulating Sepsis Treatment for ICU Patients

Sepsis is a life-threatening illness caused by the body's response to infection and a leading cause of patient mortality. Aside from different dosage of antibiotics and controlling the sources of infection, sepsis treatment involves administering intravenous fluids and vasopressors. These procedures have however shown to have drastically different results on different patients and there is a lack of efficient real-time decision support tools to guide physicians. In this project we studied the application of reinforcement learning towards learning such policies from MIMIC which is an open patient EHR dataset from ICU patients. We built a custom OpenAI Gym environment to simulate the MIMIC Sepsis cohort and ran off-the-shelf OpenAI Baselines algorithms on our custom environment. We additionally replicated the work done by Raghu et al. on Off Policy Deep RL for learning Sepsis treatment policies.


Authors: 
- Amirhossein Kiani
- Tianli Ding
- Peter Henderson


To access the simulator checkout [this repository](https://github.com/akiani/gym-sepsis).


For more information you can read our [writeup](https://github.com/akiani/rlsepsis234/blob/master/writeup.pdf) or see our [poster](https://github.com/akiani/rlsepsis234/blob/master/poster.pdf)
