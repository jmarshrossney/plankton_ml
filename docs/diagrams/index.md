---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Plankton ML - workflow diagrams
---

<!-- workaround to make diagrams responsive to dark mode -->
<style type="text/css">
svg { fill: currentColor }

path {
    fill: black;
}

@media (prefers-color-scheme: dark) {
    path { fill: white; }
}
</style>

# Workflow Diagrams


As-is and Could-be views of the flow of data from the imaging instrument to cloud-accessible storage

![From imaging instrument to cloud storage](as_is/instrument_to_store.svg) - as is
![From imaging instrument to cloud storage](could_be/instrument_to_store.svg) - as is



