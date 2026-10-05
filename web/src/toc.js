const toc = document.querySelector(".toc");
const header = document.querySelector(".site-header");
const links = Array.from(toc.querySelectorAll("a"));
const sections = links.map((link) =>
  document.getElementById(decodeURIComponent(link.hash.slice(1))),
);

function updateCurrentSection() {
  const threshold = header.getBoundingClientRect().bottom + 32;
  let current = links[0];
  sections.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= threshold) current = links[index];
  });
  for (const link of links) {
    const active = link === current;
    if (active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
  if (current) {
    const y = current.parentElement.offsetTop
      + parseFloat(getComputedStyle(current).lineHeight) / 2;
    toc.style.setProperty("--toc-marker-y", `${y}px`);
  }
}

window.addEventListener("scroll", updateCurrentSection, { passive: true });
window.addEventListener("resize", updateCurrentSection);
window.addEventListener("load", updateCurrentSection);
document.fonts.ready.then(updateCurrentSection);
updateCurrentSection();
