document.addEventListener("DOMContentLoaded", function() {
    const heroes = {{ heroes|tojson }};
    
    function updateHeroImage(selectElement, imgElementId) {
        const heroId = selectElement.value;
        const imgElement = document.getElementById(imgElementId);
        imgElement.src = heroes[heroId].image;
    }

    // Initialize images based on the default selected heroes
    for (let i = 0; i < 5; i++) {
        const radiantSelect = document.querySelector(`select[name="radiant_hero_${i}"]`);
        const direSelect = document.querySelector(`select[name="dire_hero_${i}"]`);
        updateHeroImage(radiantSelect, `radiant_hero_img_${i}`);
        updateHeroImage(direSelect, `dire_hero_img_${i}`);
    }

    window.updateHeroImage = updateHeroImage;
});
