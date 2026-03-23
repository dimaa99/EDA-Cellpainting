/* ============================================================
   Cell Painting EDA Portfolio — UI Components
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  /* ---- Collapsible sections ---- */
  document.querySelectorAll('.collapsible__toggle').forEach(toggle => {
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('open');
      const body = toggle.nextElementSibling;
      if (body) body.classList.toggle('open');
    });
  });

  /* ---- Mobile sidebar toggle ---- */
  const sidebar = document.querySelector('.sidebar');
  const btn = document.querySelector('.sidebar-toggle');
  if (btn && sidebar) {
    btn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
    // Close sidebar when clicking a nav link on mobile
    sidebar.querySelectorAll('nav a').forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 900) sidebar.classList.remove('open');
      });
    });
  }

  /* ---- Active nav highlighting (IntersectionObserver) ---- */
  const sections = document.querySelectorAll('.section[id]');
  const navLinks = document.querySelectorAll('.sidebar nav a[href^="#"]');

  if (sections.length && navLinks.length) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          navLinks.forEach(l => l.classList.remove('active'));
          const active = document.querySelector(`.sidebar nav a[href="#${entry.target.id}"]`);
          if (active) active.classList.add('active');
        }
      });
    }, { rootMargin: '-20% 0px -75% 0px' });

    sections.forEach(s => observer.observe(s));
  }

  /* ---- Lazy image loading with fade-in ---- */
  document.querySelectorAll('img[loading="lazy"]').forEach(img => {
    img.style.opacity = '0';
    img.style.transition = 'opacity 0.4s ease';
    if (img.complete) {
      img.style.opacity = '1';
    } else {
      img.addEventListener('load', () => { img.style.opacity = '1'; });
    }
  });
});
