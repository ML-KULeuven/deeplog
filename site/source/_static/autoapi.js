(function () {
  'use strict';

  window.addEventListener('DOMContentLoaded', function () {
    var pageName = '';
    if (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.pagename) {
      pageName = window.DOCUMENTATION_OPTIONS.pagename;
    }

    if (pageName.indexOf('autoapi/') === 0) {
      document.body.classList.add('is-autoapi-page');
    }
  });
})();
