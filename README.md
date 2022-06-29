**初赛任务：** 面向新闻数据的篇章级语义检索
给定一个查询，参赛系统的任务是从新闻正文集中找出与该查询语义最相关的前100篇文章，并按相关程度由高到低排序。查询的示例数据如下：

| 示例查询1 | NASA Says Saturn's Icy Moon Enceladus Could Harbor Alien Life |
| --------- | ------------------------------------------------------------ |
| 示例查询2 | S.Korean stocks edge lower, biopharma shares top drag        |

出于数据敏感性等方面考虑，所有发布数据均已进行转码处理。
因此，上述2条示例数据的实际呈现形式如下：

| 转码后示例查询1 | 39355 1205 47380 203 10109 9602 401097 731 12004 2470 325 |
| --------------- | --------------------------------------------------------- |
| 转码后示例查询2 | 1175747 18201 8181 8735 , 259500 478 1895 4498            |



思路：

- MLM预训练

- 对比学习预训练
- 对比学习有监督训练
- faiss召回