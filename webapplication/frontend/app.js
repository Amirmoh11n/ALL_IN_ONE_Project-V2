const $ = (id) => document.getElementById(id);

const dropzone = $('dropzone');
const fileInput = $('fileInput');
const preview = $('preview');
const fileName = $('fileName');
const analyzeBtn = $('analyzeBtn');

let selectedFile = null;

let lang = localStorage.getItem('neuramri.lang') || 'en';

let theme =
  localStorage.getItem('neuramri.theme') ||
  (window.matchMedia &&
  window.matchMedia('(prefers-color-scheme: light)').matches
    ? 'light'
    : 'dark');


/* =========================================================
   TRANSLATION
========================================================= */

function t(key) {
  return (
    window.NEURA_I18N?.[lang] ||
    window.NEURA_I18N.en
  )[key] || key;
}


function applyI18n() {

  document.documentElement.lang = lang;

  document.documentElement.dir =
    lang === 'fa' ? 'rtl' : 'ltr';

  document.querySelectorAll('[data-i18n]').forEach((node) => {

    const key = node.getAttribute('data-i18n');

    node.textContent = t(key);

  });

  updateLanguageButtons();

  updateStatusText();

  updateThemeButton();
}


function updateLanguageButtons() {

  const en = $('langEn');
  const fa = $('langFa');

  if (!en || !fa) return;

  en.classList.toggle('active', lang === 'en');
  fa.classList.toggle('active', lang === 'fa');
}


/* =========================================================
   THEME
========================================================= */

function applyTheme() {

  document.documentElement.classList.toggle(
    'light',
    theme === 'light'
  );

  localStorage.setItem(
    'neuramri.theme',
    theme
  );

  updateThemeButton();
}


function updateThemeButton() {

  const button = $('themeToggle');
  const icon = $('themeIcon');

  if (!button || !icon) return;

  const isLight = theme === 'light';

  button.setAttribute(
    'aria-label',
    isLight ? 'Switch to dark mode' : 'Switch to light mode'
  );

  button.setAttribute(
    'title',
    isLight ? 'Switch to dark mode' : 'Switch to light mode'
  );

  /*
   * Sun icon for light mode,
   * moon icon for dark mode.
   */
  if (isLight) {

    icon.innerHTML = `
      <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3
      6.5 6.5 0 0 0 21 12.8Z"></path>
    `;

  } else {

    icon.innerHTML = `
      <circle cx="12" cy="12" r="4"></circle>
      <path d="M12 2v2"></path>
      <path d="M12 20v2"></path>
      <path d="m4.93 4.93 1.41 1.41"></path>
      <path d="m17.66 17.66 1.41 1.41"></path>
      <path d="M2 12h2"></path>
      <path d="M20 12h2"></path>
      <path d="m6.34 17.66-1.41 1.41"></path>
      <path d="m19.07 4.93-1.41 1.41"></path>
    `;

  }
}


/* =========================================================
   ERROR HANDLING
========================================================= */

function showError(message) {

  const errorBox = $('errorBox');

  errorBox.textContent = message;

  errorBox.classList.remove('hidden');
}


function clearError() {

  $('errorBox').classList.add('hidden');

}


/* =========================================================
   FILE SELECTION
========================================================= */

function selectFile(file) {

  clearError();

  if (!file) return;

  const allowed = [
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/bmp'
  ];

  if (!allowed.includes(file.type)) {

    return showError(
      t('invalidFile')
    );

  }

  if (file.size > 10 * 1024 * 1024) {

    return showError(
      t('fileTooLarge')
    );

  }

  selectedFile = file;

  preview.src = URL.createObjectURL(file);

  fileName.textContent = file.name;

  $('emptyState').classList.add('hidden');

  $('previewState').classList.remove('hidden');

  $('resultEmpty').classList.remove('hidden');

  $('resultState').classList.add('hidden');

}


/* =========================================================
   DRAG & DROP
========================================================= */

dropzone.addEventListener('click', (e) => {

  if (
    e.target !== analyzeBtn &&
    !e.target.closest('#analyzeBtn')
  ) {
    fileInput.click();
  }

});


fileInput.addEventListener(
  'change',
  () => selectFile(fileInput.files[0])
);


['dragenter', 'dragover'].forEach((type) => {

  dropzone.addEventListener(type, (e) => {

    e.preventDefault();

    dropzone.classList.add('drop-active');

  });

});


['dragleave', 'drop'].forEach((type) => {

  dropzone.addEventListener(type, (e) => {

    e.preventDefault();

    dropzone.classList.remove('drop-active');

  });

});


dropzone.addEventListener('drop', (e) => {

  selectFile(
    e.dataTransfer.files[0]
  );

});


/* =========================================================
   LANGUAGE
========================================================= */

$('langEn').addEventListener(
  'click',
  () => {

    lang = 'en';

    localStorage.setItem(
      'neuramri.lang',
      lang
    );

    applyI18n();

  }
);


$('langFa').addEventListener(
  'click',
  () => {

    lang = 'fa';

    localStorage.setItem(
      'neuramri.lang',
      lang
    );

    applyI18n();

  }
);


/* =========================================================
   THEME BUTTON
========================================================= */

$('themeToggle').addEventListener(
  'click',
  () => {

    theme =
      theme === 'dark'
        ? 'light'
        : 'dark';

    applyTheme();

  }
);


/* =========================================================
   MODEL STATUS
========================================================= */

function updateStatusText() {

  const statusText = $('statusText');

  if (!statusText) return;

  /*
   * Preserve the current state after language switching.
   */
  const dot = $('statusDot');

  if (
    dot.classList.contains('bg-emerald-400')
  ) {

    statusText.textContent =
      t('modelReady');

  } else if (
    dot.classList.contains('bg-red-400')
  ) {

    statusText.textContent =
      t('apiDown');

  } else {

    statusText.textContent =
      t('modelMissing');

  }

}


async function checkHealth() {

  try {

    const response =
      await fetch('/api/health');

    const data =
      await response.json();

    if (data.model_ready) {

      $('statusDot').className =
        'status-dot h-2 w-2 rounded-full bg-emerald-400 text-emerald-400';

      $('statusText').textContent =
        t('modelReady');

    } else {

      $('statusDot').className =
        'status-dot h-2 w-2 rounded-full bg-amber-400 text-amber-400';

      $('statusText').textContent =
        t('modelMissing');

    }

  } catch {

    $('statusDot').className =
      'status-dot h-2 w-2 rounded-full bg-red-400 text-red-400';

    $('statusText').textContent =
      t('apiDown');

  }

}


/* =========================================================
   ANALYSIS
========================================================= */

analyzeBtn.addEventListener(
  'click',
  async (e) => {

    e.stopPropagation();

    clearError();

    if (!selectedFile) return;

    analyzeBtn.disabled = true;

    analyzeBtn.textContent =
      t('analyzing');

    try {

      const form =
        new FormData();

      form.append(
        'file',
        selectedFile
      );

      const response =
        await fetch(
          '/api/predict',
          {
            method: 'POST',
            body: form
          }
        );

      const data =
        await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          t('inferenceFailed')
        );

      }

      /*
       * SHOW RESULT
       */

      $('resultEmpty')
        .classList
        .add('hidden');

      $('resultState')
        .classList
        .remove('hidden');


      /*
       * PREDICTED CLASS
       */

      $('predictedClass')
        .textContent =
        data.predicted_class;


      /*
       * TUMOR STATUS
       */

      $('tumorSummary')
        .textContent =
        data.tumor_present
          ? t('tumorYes')
          : t('tumorNo');


      /*
       * CONFIDENCE
       */

      $('confidence')
        .textContent =
        `${data.confidence_percentage.toFixed(1)}%`;


      $('confidenceBar')
        .style
        .width =
        `${data.confidence_percentage}%`;


      /*
       * MODEL VERSION
       */

      $('modelBadge')
        .textContent =
        `${t('completed')} · v${
          data.model_version || '2.0.0'
        }`;


      /*
       * DEVICE
       */

      $('device')
        .textContent =
        (data.device || '')
          .replace(
            'ExecutionProvider',
            ''
          );


      /*
       * WARNING
       */

      $('warning')
        .textContent =
        data.warning ||
        t('disclaimer');


      /*
       * PROBABILITIES
       */

      $('probabilities').innerHTML =
        data.ranked_probabilities
          .map((item) => {

            return `
              <div class="probability-item">

                <div class="mb-1.5 flex items-center justify-between text-xs">

                  <span
                    class="capitalize"
                    style="color: var(--text-soft);"
                  >
                    ${item.class_name}
                  </span>

                  <span
                    class="font-mono"
                    style="color: var(--text-muted);"
                  >
                    ${item.percentage.toFixed(1)}%
                  </span>

                </div>

                <div
                  class="h-1.5 overflow-hidden rounded-full"
                  style="
                    background:
                      rgba(120,180,195,.10);
                  "
                >

                  <div
                    class="bar h-full rounded-full"
                    style="
                      width: ${item.percentage}%;
                      background:
                        linear-gradient(
                          90deg,
                          #168ba1,
                          #63d9e8
                        );
                    "
                  ></div>

                </div>

              </div>
            `;

          })
          .join('');

    } catch (err) {

      showError(
        err.message
      );

    } finally {

      analyzeBtn.disabled = false;

      analyzeBtn.textContent =
        t('analyze');

    }

  }
);


/* =========================================================
   INITIALIZATION
========================================================= */

applyTheme();

applyI18n();

checkHealth();