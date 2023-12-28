/**
使用jquery，把数据渲染到页面上的通用方法，支持table、dict类型

数据必须是以下格式：
表格数据：
{
     'code': 0,             <-- 返回码
     'msg': 'ok',           <-- 返回的消息，一般不用
     'title': 'QMT信息',     <-- 数据的标题
     'type': 'table',       <-- 数据的类型，目前只有：table和字典
     'data': [              <-- 表格数据，数组，每个数组里是一个字典
         {'交易1':11,'交易2':12},
         {'交易1':21,'交易2':22}
     ]
}

字典数据：
{
     'code': 0,             <-- 返回码
     'msg': 'ok',           <-- 返回的消息，一般不用
     'title': 'QMT信息',     <-- 数据的标题
     'type': 'dict',        <-- 数据的类型，目前只有：table和字典
     'data': {              <-- 字典数据
        '总资金':1234,
        '总市值':5678}
     }
}
**/

function render_html(_data,content_selector){
        // 如果有title属性，插入到html中，用H4
        if ('title' in _data){
//            console.log('小标题：',+ _data.title);
            $(content_selector).append($('<h4/>',{'style':'text-align:left'}).html(_data.title));
        }

        if (! 'type' in _data){
           console.log('错误！此行数据中无type属性，无法判断数据类型');
           return;
        }

        if (! 'data' in _data){
           console.log('错误！此行数据中无data属性，无法处理数据');
           return;
        }

        //如果是表格类型，插入表格
        if (_data.type == 'table'){
            create_table(_data.data,content_selector);
            return;
        }

        //如果是dict字典类型，插入key value表格
        if (_data.type == 'dict'){
            create_dict(_data.data,content_selector);
            return;
        }

        console.log('错误！此行数据中type属性为['+_data.type+']，无法判断数据类型，允许的类型为：table,dict');
        console.log(_data)
}

//动态创建字典类型数据展示
function create_dict(data,selector){
//    console.log('显示dict数据：'+data);
    var root = $('<ul/>')
    root.css("class","list-group");
    for (var k in data) {
        v = data[k]
        var item = $('<li/>',{"class":"list-group-item"}).html(k);
        var value = $('<span/>',{"class":"badge"}).html(v);
        item.append(value);
        root.append(item);
    }
    $(selector).append(root);
}

//动态创建表格
function create_table(list, selector) {

     var table = $('<table/>',{'class':'table table-hover table-bordered table-striped'});
     //从数组的第一行的字典里，获得表头
     var cols = get_header(list, table);
     for (var i = 0; i <list.length ; i++) {
         var row = $('<tr/>');
         for (var colIndex = 0; colIndex <cols.length ; colIndex++)
         {
             var val = list [i][cols[colIndex]];
             if (val == null) val = "" ;
             row.append($('<td/>').html(val));
         }
         table.append(row);
     }

     $(selector).append(table);
}

//动态创建表格头
function get_header(list, table) {
 var columns = [];
 var header = $('<tr />');
 for (var i = 0; i <list.length ; i++) {
     var row = list [i];
     for (var k in row) {
         if ($.inArray(k, columns) == -1) {
             columns.push(k);
             header.append($('<th/>').html(k));
         }
     }
 }
 table.append(header);
 return columns;
}