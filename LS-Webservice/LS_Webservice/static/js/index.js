function addDrug() {
    var seqNodes = $(".sequence");
    var seq = seqNodes.length + 1;
    var html = $('<div class="drug-profile"></div>');
    var sequenceInput = '<div class="form-row"><div class="form-group col-md-1" id="seq"><label for="sequence' +  seq + '">Seq.</label><input id="sequence-' + seq  + '" class="form-control sequence" name="sequence" value="' + seq + '" /> </div>';
    var rxcuiInput = '<div class="form-group col-md-2" id="rxcui"><label for="rxcui-input-'+ seq + '">Rxcui</label> <input type="text" class="form-control" id="rxcui-input-'+ seq + '" name="rxcui"/></div>';
    var reporedRoleInput = '<div class="form-group col-md-3"><label for="role-select-' +  seq + '">Reported Role</label> <select class="form-control" name="reported-role" id="role-select-' + seq + '"><option value="">Unknown</option><option value="PS">(PS) Primary Suspect Drug</option> <option value="SS">(SS) Secondary Suspect Drug</option> <option value="C">(C) Concomitant</option> <option value="I">(I) Interacting</option> </select> </div>';
    var dechalInput = '<div class="form-group col-md-3"><label for="dechal-select-' + seq + '">Dechallenge Code</label><select class="form-control" name="dechal" id="dechal-select-' + seq + '"><option value="">Unknown</option><option value="Y">(Y) Positive Dechallenge</option> <option value="N">(N) Negative Dechallenge</option> <option value="U">(U) Unknown</option> <option value="D">(D) Does Not Apply</option> </select> </div>';
    var rechalInput = '<div class="form-group col-md-3"> <label for="rechal-select-' + seq + '">Rechallenge Code</label> <select class="form-control" name="rechal" id="rechal-select-' + seq + '"><option value="">Unknown</option><option value="Y">(Y) Positive Rechallenge</option> <option value="N">(N) Negative Rechallenge</option> <option value="U">(U) Unknown</option> <option value="D">(D) Does Not Apply</option> </select> </div> </div>';
    var indicationInput = '<div class="form-row"> <div class="form-group col-md-5"> <label for="indication-input-' + seq + '">Indications</label> <input class="form-control" type="text" name="indications" id="indication-input-' + seq + '" /> </div> </div>';
    var inputs = sequenceInput + rxcuiInput + reporedRoleInput + dechalInput + rechalInput + indicationInput;
    html.html(inputs);
    $("#drug-profile").append(html);
    $.get(window.location.origin + "/getRxcuis")
            .done(function(data) {
            $( "#rxcui-input-" + seq).autocomplete({
                source: data,
                minLength: 3
            });
      });

    $.get(window.location.origin + "/getIndications")
            .done(function(data) {
            $( "#indication-input-" + seq).autocomplete({
                source: data,
                minLength: 3
            });
      });
}

$(document).ready(function($) {
      $.get(window.location.origin + "/getRxcuis")
            .done(function(data) {
            $( "#rxcui-input-1" ).autocomplete({
                source: data,
                minLength: 3
            });
      });

    $.get(window.location.origin + "/getIndications")
            .done(function(data) {
            $( "#indication-input-1" ).autocomplete({
                source: data,
                minLength: 3
            });
      });
});