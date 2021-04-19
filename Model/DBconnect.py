import pymysql
import datetime


class DBconnect:
    """
    数据库连接类，初始化将会连接数据库，其中的方法是增删改查，析构将会关闭数据库连接

    Attributes:
        None

    """

    def __init__(self):
        """
        初始化数据库链接地址，连接数据库
        """
        try:
            self.conn = pymysql.connect(
                host='127.0.0.1', port=3306, user='root', passwd="123456", db='animeface'
            )
            self.cur = self.conn.cursor()
        except e:
            print(e)

    # 查询
    def dbQuery(self, dbTable):
        """
        数据库查询代码，将需要查询的表名传入

        Args:
            dbTable: 需要查询的表名

        Returns:
            一个查询结果的List，没有任何数据过滤，存粹”SELECT * FROM “返回表
        """
        cur = self.cur
        sql = "SELECT * FROM "+dbTable
        cur.execute(sql)
        returnList = []
        for r in cur:
            returnList.append(r)
            # print(r)
        return returnList


    def dbQuery_title_len(self, dbTable):
        """
        查询表中长度

        Args:
            dbTable 需要查询的表

        Returns:
            返回一个数字，即拥有多少已经存储的内容
        """
        cur = self.cur
        sql = "select  count(*) from `"+dbTable+"`"
        print(sql)
        cur.execute(sql)
        for r in cur:
            return r[0]

    def dbQuery_userLogin(self, user_name, user_pwd):
        """
        通过用户名密码进行登陆判断，准备改成使用用户账户名和密码登陆的方式
        Update:
            已修改为用户名和密码的登录方式。

        Args:
            user_name 用户名
            user_pwd 用户密码

        Returns:
            返回一个查询结果，即存在该用户ID和其对应的密码则为登陆成功，否则为失败。
        """
        conn = self.conn
        cur = self.cur
        dbTable = "user_info"
        sql = "SELECT * FROM "+dbTable+" WHERE userName='" + \
            user_name+"' and userPwd='"+user_pwd+"'"
        print(sql)
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            print("操作异常：%s" % str(e))
            # 错误回滚
            conn.rollback()
            return e
        # 返回第一个合适的信息 - 也只有一个合适的信息
        for r in cur:
            return r


    def dbDelete(self, dbTable, needId, inputId):
        """
        删除表中特定字段以及对应该字段的值的记录
        Args:
            needId 需要查询的字段，比如说user_id
            inputId 需要删除该对应字段的记录，比如说 momon1

        """
        conn = self.conn
        cur = self.cur
        sql = "DELETE from "+dbTable+" where "+needId+"="+inputId
        #sql = 'DELETE from user_info where userId='+inputId
        print(sql)
        try:
            cur.execute(sql)
            conn.commit()
            return True
        except Exception as e:
            print("操作异常：%s" % str(e))
            # 错误回滚
            conn.rollback()
            return False

    def dbInsert(self, dbTable, *args):
        """
        插入数据库代码，根据表名称自动产生对应该表名称的插入代码，
        但是前提是传入的args的值必须合适，不能多也不能少

        Args:
            dbTable 数据表
            args 多参数传入值，必须要和数据库字段一一对应
        """
        print("------------DBINSERT-------------")
        conn = self.conn
        cur = self.cur

        # python没有switch，本身switch需要哈希比较的，这和Python倡导的灵活性相互驳斥，反而会退化到IF-ELIF-ELSE级别
        # 所以就用if-elif-else进行特判表对应的sql语句
        sql = ""
        print(args)
        if dbTable == "user":
            sql = "INSERT INTO "+dbTable+" VALUES(%s,%s,%s,%s,%s);"
        elif dbTable == "savepath":
            sql = "INSERT INTO "+dbTable+" VALUES(%s,%s,%s,%s);"
        print(sql)
        try:
            cur.execute(sql, args)
            conn.commit()
            print("insert successful!")
        except Exception as e:
            print("操作异常：%s" % str(e))
            # 错误回滚
            conn.rollback()
            return False
        return True

            

    # 测试更新、修改代码 - 完成
    # 封装更新，修改代码 - 完成
    # dbTable 表名称 -  needValue 需要修改的值名 - inputValue 需要修改的值 - needId 查询的ID名 - inputId 查询的ID具体内容
    def dbUpdate_signled(self, dbTable, needValue, inputValue, needId, inputId):
        """
        对数据库中单个表的参数进行修改，
        Args：
            needValue 需要修改的值名 
            inputValue 需要修改的值 
            needId 查询的ID名 
            inputId 查询的ID具体内容
        """
        conn = self.conn
        cur = self.cur
        sql = "update " + dbTable+" set "+needValue+"=\'" + \
            inputValue+"\' where " + needId + "=" + inputId
        #sql = "update user_info set userPwd='666666' where userId='1111'"
        print(sql)
        try:
            cur.execute(sql)
            conn.commit()
            return True
        except Exception as e:
            print("操作异常：%s" % str(e))
            # 错误回滚
            conn.rollback()
            return False

    

    def __del__(self):
        """
        析构函数，关闭表和查询
        """
        self.cur.close()
        self.conn.close()


if __name__ == '__main__':
    db = DBconnect()

    #chooseTable = "user"
    #k = db.dbInsert(chooseTable,"10000009","common","123123","common@qq.com","18278365689")
    chooseTable = "savepath"
    k = db.dbInsert(chooseTable,"00000001","Face/user/10000001/random","123123","common@qq.com")
    
    print(k)
    # db.dbQuery(chooseTable)
