$(document).ready(function($) {
      $.get(window.location.origin + "/getRxcuis")
            .done(function(data) {
                var stringArr = data.map(String);
                  $('#rxcui-tag1').tagit({
                        fieldName: "rxcui1",
                        autocomplete: {delay: 0, minLength: 3},
                        availableTags: stringArr,
                        singleField: true
                  });
                  $('#rxcui-tag2').tagit({
                        fieldName: "rxcui2",
                        autocomplete: {delay: 0, minLength: 3},
                        availableTags: stringArr,
                        tagLimit:1
                  });
            });


});