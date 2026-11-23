# Scene diversity generation

Interactive mug-and-spoon demonstration: physical and particle coordinates, recorded MuJoCo releases, material-point scene distance, objective landscapes and family galleries.

Live demo: [c0rmac.github.io/scene-diversity-generation](https://c0rmac.github.io/scene-diversity-generation/).

The release player uses 18 recorded Q13 starts and one separate initial-overlap example, with motion/penetration annotations and slow-motion controls. The metric illustration includes the fixed body sample sets.

A second page, [Consensus with births and deaths over contact types](https://c0rmac.github.io/scene-diversity-generation/solver_bd372/), is in `solver_bd372/`: the collar formulation with a Fisher–Rao birth–death step at friction 0.6, with its own interactive figures and a gallery by tested component.

A third page, [A search space of touching scenes](https://c0rmac.github.io/scene-diversity-generation/contact_pair_space/), is in `contact_pair_space/`. The code that builds that search space, with tests and a quick start, is in [`contact_pair_space/code/`](contact_pair_space/code/).

This is a static site. To preview locally, run `python3 -m http.server 8000` in this directory. GitHub Pages serves `main` from the repository root.

The page describes diagnostic study arm Q13. Coarse contact families and finite release checks are reported with their limitations on the page.
