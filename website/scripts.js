// Toggle mobile menu
const burger = document.querySelector('.burger');
const navLinks = document.querySelector('.nav-links');

burger.addEventListener('click', () => {
    navLinks.classList.toggle('active');

    // Animate burger lines
    burger.classList.toggle('toggle');
});

// Auto-Rotating Carousel
const track = document.querySelector('.carousel-track');
const slides = Array.from(track.children);
let currentIndex = 0;

const autoRotate = () => {
    currentIndex = (currentIndex + 1) % slides.length;
    const width = slides[0].getBoundingClientRect().width;
    track.style.transform = `translateX(-${currentIndex * width}px)`;
};

// Start auto-rotation
setInterval(autoRotate, 3000);