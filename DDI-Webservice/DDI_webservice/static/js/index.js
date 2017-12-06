$(document).ready(function($) {
      $.get(window.location.origin + "/getDrugs")
            .done(function(data) {
                var stringArr = data.map(String);
                  $('#drug-tag1').tagit({
                        fieldName: "drug1",
                        autocomplete: {delay: 0, minLength: 2},
                        availableTags: stringArr,
                        singleField: true
                  });
                  $('#drug-tag2').tagit({
                        fieldName: "drug2",
                        autocomplete: {delay: 0, minLength: 2},
                        availableTags: stringArr,
                        tagLimit:1
                  });
            });


});